/*
 * Copyright Notice:
 * -----------------
 *
 * The contents of this file are subject to the PfTijah Public License
 * Version 1.1 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 * http://dbappl.cs.utwente.nl/Legal/PfTijah-1.1.html
 *
 * Software distributed under the License is distributed on an "AS IS"
 * basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
 * License for the specific language governing rights and limitations
 * under the License.
 * 
 * The Original Code is the Mirex system.
 * 
 * The Initial Developer of the Original Code is the "University of Twente".
 * Portions created by the "University of Twente" are
 * Copyright (C) 2010 "University of Twente".
 * All Rights Reserved.
 */

package nl.utwente.mirex;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.security.InvalidParameterException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Scanner;

import nl.utwente.mirex.util.KeyValueInputFormat;
import nl.utwente.mirex.util.WarcTextConverterInputFormat;

import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.StringUtils;

/**
 * <b>Runs MapReduce job:</b> Runs a simple TREC experiment. 
 * The input consists of gzipped, tab separated files containing: 
 * <i>WARC-TREC-ID, URL, anchor text1, anchor text 2, </i>
 * (i.e., the result of AnchorExtract.java)
 * and a TREC query file formatted as
 * <i> query-id ":" query text, </i>
 * (i.e., the query file provided by NIST).
 * This MapReduce program is described in: 
 * <blockquote>
 *   Djoerd Hiemstra and Claudia Hauff. 
 *   "MIREX: MapReduce Information Retrieval Experiments" 
 *   Technical Report TR-CTIT-10-15, Centre for Telematics 
 *   and Information Technology, University of Twente, 
 *   ISSN 1381-3625, 2010
 * </blockquote>
 * @author Djoerd Hiemstra
 * @author Michael Meijer
 * @since 0.1
 * @see AnchorExtract
 */
public class TrecRun {
	private static org.apache.commons.logging.Log Log = org.apache.commons.logging.LogFactory.getLog("org.apache.hadoop.mapred.Task");
   /**
    * -- Mapper: Runs all queries on one document. 
    */
	public static class Map extends Mapper<Text, Text, Text, Text> {

     private static final String TOKENIZER = "[^0-9A-Za-z]+";
     private java.util.Map<String, String[]> trecQueries = new HashMap<String, String[]>();
     private java.util.Map<String, Integer> queryTerms = new HashMap<String, Integer>();
     
     public void setup(Context context) {
       Path[] queryFiles;
       try {
         queryFiles = DistributedCache.getLocalCacheFiles(context.getConfiguration());
         parseQueryFile(queryFiles[0]);
       } catch (IOException ioe) {
    	   Log.error("Error reading query file "+StringUtils.stringifyException(ioe));
    	   System.exit(1);
       }
     }

     private void parseQueryFile(Path queryFile) throws IOException {
       BufferedReader fis = new BufferedReader(new FileReader(queryFile.toString()));
       String queryString = null;

       while ((queryString = fis.readLine()) != null) {
         if (queryString.startsWith("#MIREX")) throw new IOException("Wrong format, use original TREC topic format.");
         queryString = queryString.toLowerCase();
         String [] fields = queryString.split(":");
         String [] terms = fields[1].split(TOKENIZER);
         trecQueries.put(fields[0], terms);
         for (int i=0; i < terms.length; i++) {
           queryTerms.put(terms[i], 1);
         }
       }
       Log.info("Using "+trecQueries.size()+" queries");
     }

     private Double scoreDocumentLM(String[] qterms, java.util.Map<String, Integer> docTF, Integer doclength) {
       Double score = 1.0d;
       for (int i=0; i < qterms.length; i++) {
         Integer tf = (Integer) docTF.get(qterms[i]);
         if (tf != null) score *= (new Double(tf) / new Double(doclength));
         else return 0.0d;  // no match
       }
       return score * doclength; // length prior
     }

     /**
      * @param key TREC-ID
      * @param value document text
      * @param output (Query-ID, TREC-ID, score)
      */
     public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
       // Store tf's of document only for term that is in one of the queries
       java.util.Map<String, Integer> docTF = new HashMap<String, Integer>();
       Integer doclength = 0; // one more, so at least 1.
       Scanner scan = new Scanner(value.toString().toLowerCase()).useDelimiter(TOKENIZER);
       while (scan.hasNext()) {
         doclength++;
         String term = scan.next();
         if (queryTerms.containsKey(term)) {
           Integer freq = (Integer) docTF.get(term);
           if (freq != null) docTF.put(term, freq + 1);
           else docTF.put(term, 1);
         }
       }

       // for each query, score the document
       if (doclength > 0) {
         Iterator<String> iterator = trecQueries.keySet().iterator();
         while (iterator.hasNext()) {
           String qid = (String) iterator.next();
           String [] qterms = (String []) trecQueries.get(qid);
           Double score = scoreDocumentLM(qterms, docTF, doclength);
           if (score != 0.0d) {        	   
             context.write(new Text(qid), new Text(key.toString() + "\t" + score.toString()));
           }
         }
       }
     }
   }

   /**
    * -- Reducer: Sorts the retrieved documents and takes the top 1000.
    */
	public static class Reduce extends Reducer<Text, Text, Text, Text> {

     private static final Integer TOP = 1000;

     /**
      * @param key Query-ID
      * @param values (TREC-ID, score)
      * @param output (Query-ID, TREC-ID, score)
      */
     public void reduce(Text key, Iterator<Text> values, Context context) throws InterruptedException, IOException {

       String[] RankedQuerId = new String[TOP], RankedResult = new String[TOP];
       Double[] RankedScores = new Double[TOP];
       Integer newIndex = 0;
       Double lastScore = 0.0d; // does not matter what...
       while(values.hasNext()) {
         String value = values.next().toString();
         String[] fields = value.split("\t");
         Double score = new Double(fields[1]);
         if (newIndex < TOP || score > lastScore) {
           if (newIndex < TOP) newIndex++;
           Integer index = newIndex - 1;
           while(index > 0 && RankedScores[index-1] < score) {
             RankedScores[index] = RankedScores[index-1];
             RankedQuerId[index] = RankedQuerId[index-1]; 
             RankedResult[index] = RankedResult[index-1];
             index--;
           }
           RankedScores[index] = score;
           RankedQuerId[index] = key.toString();
           RankedResult[index] = value;
           lastScore = RankedScores[newIndex-1];
         }
       }
       for (Integer i = 0; i < newIndex; i++) {
         context.write(new Text(RankedQuerId[i]), new Text(RankedResult[i]));
       }
     }
   }


   /**
    * Runs the MapReduce job "trec run"
    * @param args 0: path to parsed document collection (use AnchorExtract); 1: (non-existing) path that will contain run results; 2: TREC query file
    * @usage. see README.html 
    */
   public static void main(String[] args) throws Exception {

	 if (args.length!=3  && args.length!=4) {
		System.out.printf( "Usage: %s [inputFormat] inputFiles outputFile topicFile \n", TrecRun.class.getSimpleName());
		System.out.println("          inputFormat: either WARC or KEYVAL; default WARC");			
		System.out.println("          inputFiles: the WARC files");
		System.out.println("          outputFiles: output directory");
		System.out.println("          topicFile: topic descriptions (one query per line)");
		System.exit(1);
	 }
	 int argc = 0;
	 String inputFormat = "WARC";
	 if (args.length>3) {
	  	 inputFormat = args[argc++].toUpperCase(); 
	 }	 	   
	 String inputFiles = args[argc++];
	 String outputFile = args[argc++];
	 String topicFile = args[argc++];
     // Set job configuration
     Job job = new Job();
     job.setJobName("MirexTrecRun");
     job.setJarByClass(TrecRun.class);
		
     // Set intermediate output (override defaults)
     job.setMapOutputKeyClass(Text.class);
     job.setMapOutputValueClass(Text.class);

     // Set output (override defaults)
     job.setOutputKeyClass(Text.class);
     job.setOutputValueClass(Text.class);


     // Set map-reduce classes
     job.setMapperClass(Map.class);
     job.setCombinerClass(Reduce.class);
     job.setReducerClass(Reduce.class);

     // Set input-output format
     if (inputFormat.equals("KEYVAL")) {
    	 job.setInputFormatClass(KeyValueInputFormat.class);
     }
     else if (inputFormat.equals("WARC")) {
    	 job.setInputFormatClass(WarcTextConverterInputFormat.class);
     }
     else {
    	 throw new InvalidParameterException("inputFormat must bei either WARC or KEYVAL");
     }
     job.setOutputFormatClass(TextOutputFormat.class);
     //job.setBoolean("mapred.output.compress", false);
		
     // Set input-output paths
     FileInputFormat.setInputPaths(job, new Path(inputFiles));
     FileOutputFormat.setOutputPath(job, new Path(outputFile));
		
     // Set job specific distributed cache file (query file)
     DistributedCache.addCacheFile(new Path(topicFile).toUri(), job.getConfiguration());
	
     // Run the job
     job.waitForCompletion(true);
   }

}
