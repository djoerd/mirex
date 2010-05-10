/*
 *
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
 * 
 * Author(s): Djoerd Hiemstra 
 *            Michael Meijer
 * 
 * About:
 * ------
 * 
 * Do a simple TREC run.
 *  Input: (argument 1) Document representation (WARC-TREC-ID, text), 
 *         tab separated
 *         (argument 3) TREC ClueWeb queries (TREC-QUERY-ID, Query terms), 
 *         separated by a colon (":")
 *  Output: (argument 2) (TREC-QUERY-ID, WARC-TREC-ID, score), tab separated 
 * 
 * Djoerd Hiemstra, February 2010
 */

package nl.utwente.mirex;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;
import java.util.HashMap;
import java.util.Scanner;
import java.lang.Math;

import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.KeyValueTextInputFormat;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.util.StringUtils;

public class TrecRun {

   public static class Map extends MapReduceBase implements Mapper<Text, Text, Text, Text> {

     private static final String TOKENIZER = "[^0-9A-Za-z]+";
     private java.util.Map<String, String[]> trecQueries = new HashMap<String, String[]>();
     private java.util.Map<String, Integer> queryTerms = new HashMap<String, Integer>();

     public void configure(JobConf job) {
       Path[] queryFiles;
       try {
         queryFiles = DistributedCache.getLocalCacheFiles(job);
         parseQueryFile(queryFiles[0]);
       } catch (IOException ioe) {
         System.err.println(StringUtils.stringifyException(ioe));
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

     public void map(Text key, Text value, OutputCollector<Text, Text> output, Reporter reporter) throws IOException {

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
         Iterator iterator = trecQueries.keySet().iterator();
         while (iterator.hasNext()) {
           String qid = (String) iterator.next();
           String [] qterms = (String []) trecQueries.get(qid);
           Double score = scoreDocumentLM(qterms, docTF, doclength);
           if (score != 0.0d) 
             output.collect(new Text(qid), new Text(key.toString() + "\t" + score.toString()));
         }
       }
     }
   }

   public static class Reduce extends MapReduceBase implements Reducer<Text, Text, Text, Text> {

     private static final Integer TOP = 1000;

     public void reduce(Text key, Iterator<Text> values, OutputCollector<Text, Text> output, final Reporter reporter) throws IOException {

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
         output.collect(new Text(RankedQuerId[i]), new Text(RankedResult[i]));
       }
     }
   }


   public static void main(String[] args) throws Exception {
     // Set job configuration
     JobConf conf = new JobConf(TrecRun.class);
     conf.setJobName("MirexTrecRun");
		
     // Set intermediate output (override defaults)
     conf.setMapOutputKeyClass(Text.class);
     conf.setMapOutputValueClass(Text.class);

     // Set output (override defaults)
     conf.setOutputKeyClass(Text.class);
     conf.setOutputValueClass(Text.class);

     // Set map-reduce classes
     conf.setMapperClass(Map.class);
     conf.setCombinerClass(Reduce.class);
     conf.setReducerClass(Reduce.class);

     // Set input-output format
     conf.setInputFormat(KeyValueTextInputFormat.class);
     conf.setOutputFormat(TextOutputFormat.class);
     conf.setBoolean("mapred.output.compress", false);
		
     // Set input-output paths
     FileInputFormat.setInputPaths(conf, new Path(args[0]));
     FileOutputFormat.setOutputPath(conf, new Path(args[1]));
		
     // Set job specific distributed cache file (query file)
     DistributedCache.addCacheFile(new Path(args[2]).toUri(), conf);
	
     // Run the job
     JobClient.runJob(conf);
   }

}
