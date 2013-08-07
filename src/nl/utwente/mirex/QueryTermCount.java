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
import java.io.InputStreamReader;
import java.security.InvalidParameterException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Scanner;

import nl.utwente.mirex.util.WarcTextConverterInputFormat;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.StringUtils;

import nl.utwente.mirex.util.WarcTextConverterInputFormat;

/**
 * <b>Runs MapReduce job:</b> Gets global statistics for query file.
 * Very similar to the famous "word count" job.
 *  Inputs: (argument 1) Document representation (WARC-TREC-ID, text), 
 *         tab separated
 *         (argument 2) TREC ClueWeb queries (TREC-QUERY-ID, Query terms), 
 *         separated by a colon (":")
 *  Output: (argument 3) TREC ClueWeb queries (TREC-QUERY-ID, Query terms),
 *         separated by a colon (":"), augmented with global statistics
 * 
 * @author Djoerd Hiemstra
 * @since 0.2
 * @see AnchorExtract
 */
public class QueryTermCount {

   private static final String SysName = "MIREX";
   private static final String CollectionLength = SysName + "-LENGTH";
   private static final String NumberOfDocs = SysName + "-NDOCS";
   private static final String DF = SysName + "-DF-";
   private static final String CF = SysName + "-CF-";
   private static final String tempName = SysName + "-tmp";
   private static final String TOKENIZER = "[^0-9A-Za-z]+";

  /**
  * -- Mapper: Collects local statistics for one document. 
  */
  public static class Map extends Mapper<Text, Text, Text, LongWritable> {

     private static final LongWritable one = new LongWritable(1);  

     private java.util.Map<String, String[]> trecQueries = new HashMap<String, String[]>();
     private java.util.Map<String, Integer> queryTerms = new HashMap<String, Integer>();

     public void setup(Context context) {
       Path[] queryFiles;
       try {
         queryFiles = DistributedCache.getLocalCacheFiles(context.getConfiguration());
         parseQueryFile(queryFiles[0]);
       } catch (IOException ioe) {
         System.err.println(StringUtils.stringifyException(ioe));
         System.exit(1);
       }
     }

     private void parseQueryFile(Path queryFile) {
       try {
         BufferedReader fis = new BufferedReader(new FileReader(queryFile.toString()));
         String query = null;

         while ((query = fis.readLine()) != null) {
           String [] fields = query.toLowerCase().split(":");
           String [] terms = fields[1].split(TOKENIZER);
           trecQueries.put(fields[0], terms);
           for (int i=0; i < terms.length; i++) {
             queryTerms.put(terms[i], 1);
           }
         }
       } catch (IOException ioe) {
         System.err.println(StringUtils.stringifyException(ioe));
         System.exit(1);
       }
     }

     /**
      * @param key TREC-ID
      * @param value document text
      * @param output (Query-term <i>or</i> intermediate statistic, Count)
      */
     public void map(Text key, Text value, Context context) throws IOException, InterruptedException {

       // Store tf's of document only for term that is in one of the queries
       java.util.Map<String, Integer> docTF = new HashMap<String, Integer>();
       Long doclength = 0l;
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
       context.write(new Text(CollectionLength), new LongWritable(doclength));
       context.write(new Text(NumberOfDocs), one);
       Iterator<String> iterator = docTF.keySet().iterator();
       while (iterator.hasNext()) {
         String term = (String) iterator.next();
         Integer count = (Integer) docTF.get(term);
         context.write(new Text(CF + term), new LongWritable(count));
         context.write(new Text(DF + term), one);
       }
     }
   }

  /**
  * -- Reducer: Sums all statistics. 
  */
  public static class Reduce extends Reducer<Text, LongWritable, Text, LongWritable> {

     /**
      * @param key Query-term <i>or</i> intermediate statistic
      * @param values Counts
      * @param output (Query-term <i>or</i> intermediate statistic, Summed count)
      */
		public void reduce(Text key, Iterable<LongWritable> values,
				Context context) throws InterruptedException, IOException {

       Long sum = 0l;       
       for (LongWritable val: values) {
         sum += val.get();
       }
       context.write(key, new LongWritable(sum));
     }
   }

   /**
   * Configure the Hadoop job
 * @throws IOException 
   */
   public static Job configureJob (String jobName, String format, Path inputFile, Path tempOut, Path topicFile) throws IOException, InvalidParameterException {
     // Set job configuration
     Job job = new Job();
     job.setJobName(jobName);
     job.setJarByClass(QueryTermCount.class);

     // Set intermediate output (override defaults)
     job.setMapOutputKeyClass(Text.class);
     job.setMapOutputValueClass(LongWritable.class);

     // Set output (override defaults)
     job.setOutputKeyClass(Text.class);
     job.setOutputValueClass(LongWritable.class);

     // Set map-reduce classes
     job.setMapperClass(Map.class);
     job.setCombinerClass(Reduce.class);
     job.setReducerClass(Reduce.class);

     // Set input-output format
     if (format.equals("KEYVAL")) {
    	 job.setInputFormatClass(KeyValueTextInputFormat.class);
     }
     else if (format.equals("WARC")) {
    	 job.setInputFormatClass(WarcTextConverterInputFormat.class);
     }
     else {
    	 throw new InvalidParameterException("inputFormat must bei either WARC or KEYVAL");
     }
	 job.setOutputFormatClass(TextOutputFormat.class);
     // also works withoput
     //conf.set("mapred.output.compress", false);
     job.setNumReduceTasks(1);

     // Set input-output paths
     FileInputFormat.setInputPaths(job, inputFile);
     FileOutputFormat.setOutputPath(job, tempOut);

     // Set job specific distributed cache file (query file)
     DistributedCache.addCacheFile(topicFile.toUri(), job.getConfiguration());

     return job ;
   } 

 /**
  * Runs the MapReduce job that gets global statistics
  * @param args 0: path to parsed document collection (use AnchorExtract); 1: TREC query file; 2: MIREX query file with global statistics
  * @usage. 
  * <code> % hadoop jar mirex-0.2.jar nl.utwente.mirex.QueryTermCount  WARC warc wt2010-topics.stats wt2010-topics.queries-only  </code> 
  */
  public static void main(String[] args) throws Exception {
	 if (args.length!=3  && args.length!=4) {
		System.out.printf( "Usage: %s [inputFormat] inputFiles topicFile outputFile\n", QueryTermCount.class.getSimpleName());
		System.out.println("          inputFormat: either WARC or KEYVAL; default WARC");
		System.out.println("          inputFiles: path to data");
		System.out.println("          outputFile: topic file with statistics");
		System.out.println("          topicFile: topic file in format queryId: term1 term2...");					
		System.exit(1);
	 }
	  
     Path tempOut = new Path(tempName);   
     int argc=0;
     String inputFormat = "WARC";
     if (args.length>3) {
    	 inputFormat = args[argc++]; 
     }
     Path inputFile = new Path(args[argc++]);
     Path topicFile = new Path(args[argc++]);
     Path outputFile = new Path(args[argc++]);
     
     java.util.Map<String, Long> queryCounts = new HashMap<String, Long>();
		
     // Stop if out file exists
     FileSystem hdfs = FileSystem.get(new Configuration());
     if (hdfs.exists(outputFile)) {
       System.err.println("Output file " + outputFile + " already exists.");
       System.exit(1); 
     }
     hdfs.delete(tempOut, true);

     // Run the job
     Job job = configureJob("QueryTermCount", inputFormat, inputFile, tempOut, topicFile);
     job.waitForCompletion(true);
 
     // Get created global statistics from all files which start with "part" from tempOut
     try {
         String tempLine;
         FileStatus[] status = hdfs.listStatus(tempOut);       
         for (int i=0;i<status.length;i++){
      	   String fileName = status[i].getPath().getName();
      	   if (!fileName.startsWith("part")) continue;      	   
  	       FSDataInputStream dis = hdfs.open(status[i].getPath());
  	       //BufferedReader in = new BufferedReader();
  	       BufferedReader in= new BufferedReader(new InputStreamReader(dis));
  	       while ((tempLine = in.readLine()) != null) { 
  	    	 String[] fields = tempLine.split("\t");
  	         queryCounts.put(fields[0], new Long(fields[1]));
  	       }
  	       dis.close();
         }
       } catch (IOException ioe) {
         System.err.println(StringUtils.stringifyException(ioe));
         System.exit(1);
       }

     // Write new topic file with global statistics
     try {
       String tempLine;
       FSDataOutputStream dos = hdfs.create(outputFile);
       dos.writeBytes("#MIREX-COMMENT: query term weight, document frequency, collection frequency (for each term)\n"); 
       dos.writeBytes("#MIREX-COLLECTION:" + inputFile + "\n");
       dos.writeBytes("#" + CollectionLength + ":" + queryCounts.get(CollectionLength) + "\n");
       dos.writeBytes("#" + NumberOfDocs + ":" + queryCounts.get(NumberOfDocs) + "\n");

       FSDataInputStream dis = hdfs.open(topicFile);
       BufferedReader in= new BufferedReader(new InputStreamReader(dis));
       while ((tempLine = in.readLine()) != null) {
         String [] fields = tempLine.toLowerCase().split(":");
         dos.writeBytes(fields[0] + ":");
         String [] terms = fields[1].replaceAll("=", " ").split(TOKENIZER);
         for (int i=0; i < terms.length; i++) {
           Long df, cf;
           if (queryCounts.containsKey(DF + terms[i]))  {
             df = queryCounts.get(DF + terms[i]); 
             cf = queryCounts.get(CF + terms[i]);
           }
           else {
             df = 0l;
             cf = 0l;
           }
           dos.writeBytes(terms[i] + "=1=" + df.toString() + "=" + cf.toString());
           if (i < terms.length - 1) dos.writeBytes(" "); 
         }
         dos.writeBytes("\n");
      }
      dis.close();
      dos.close();
    } catch (IOException ioe) {
      System.err.println(StringUtils.stringifyException(ioe));
      System.exit(1);
    }
    hdfs.close();
  }

}
