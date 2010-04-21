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
 * 
 * About:
 * ------
 * 
 * Do a simple TREC run.
 *  Input: (argument 1) Document representation (WARC-TREC-ID, text), 
 *         tab separated
 *         (argument 2) TREC ClueWeb queries (TREC-QUERY-ID, Query terms), 
 *         separated by a colon (":")
 *  Output: (argument 3) TREC ClueWeb queries (TREC-QUERY-ID, Query terms),
 *         separated by a colon (":"), augmented with global statistics
 * 
 * Djoerd Hiemstra, April 2010
 */

package nl.utwente.mirex;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;
import java.util.HashMap;
import java.util.Scanner;
import java.lang.Math;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem; 
import org.apache.hadoop.fs.FSDataOutputStream; 
import org.apache.hadoop.fs.FSDataInputStream; 
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.LongWritable;
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

public class QueryTermCount {

   private static final String SysName = "MIREX";
   private static final String CollectionLength = SysName + "-LENGTH";
   private static final String NumberOfDocs = SysName + "-NDOCS";
   private static final String DF = SysName + "-DF-";
   private static final String CF = SysName + "-CF-";
   private static final String tempName = SysName + "-tmp";
   private static final String TOKENIZER = "[^0-9A-Za-z]+";

   public static class Map extends MapReduceBase implements Mapper<Text, Text, Text, LongWritable> {

     private static final LongWritable one = new LongWritable(1);  

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

     private void parseQueryFile(Path queryFile) {
       try {
         BufferedReader fis = new BufferedReader(new FileReader(queryFile.toString()));
         String query = null;

         while ((query = fis.readLine()) != null) {
           query.toLowerCase();
           String [] fields = query.split(":");
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

     public void map(Text key, Text value, OutputCollector<Text, LongWritable> output, Reporter reporter) throws IOException {

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
       output.collect(new Text(CollectionLength), new LongWritable(doclength));
       output.collect(new Text(NumberOfDocs), one);
       Iterator iterator = docTF.keySet().iterator();
       while (iterator.hasNext()) {
         String term = (String) iterator.next();
         Integer count = (Integer) docTF.get(term);
         output.collect(new Text(CF + term), new LongWritable(count));
         output.collect(new Text(DF + term), one);
       }
     }
   }

   public static class Reduce extends MapReduceBase implements Reducer<Text, LongWritable, Text, LongWritable> {

     public void reduce(Text key, Iterator<LongWritable> values, OutputCollector<Text, LongWritable> output, final Reporter 
reporter) throws IOException {

       Long sum = 0l;
       while (values.hasNext()) {
         sum += values.next().get();
       }
       output.collect(key, new LongWritable(sum));
     }
   }

   public static JobConf configureJob (String jobName, Path inputFile, Path tempOut, Path topicFile) {
     // Set job configuration
     JobConf conf = new JobConf(TrecRun.class);
     conf.setJobName(jobName);

     // Set intermediate output (override defaults)
     conf.setMapOutputKeyClass(Text.class);
     conf.setMapOutputValueClass(LongWritable.class);

     // Set output (override defaults)
     conf.setOutputKeyClass(Text.class);
     conf.setOutputValueClass(LongWritable.class);

     // Set map-reduce classes
     conf.setMapperClass(Map.class);
     conf.setCombinerClass(Reduce.class);
     conf.setReducerClass(Reduce.class);

     // Set input-output format
     conf.setInputFormat(KeyValueTextInputFormat.class);
     conf.setOutputFormat(TextOutputFormat.class);
     conf.setBoolean("mapred.output.compress", false);
     conf.setNumReduceTasks(1);

     // Set input-output paths
     FileInputFormat.setInputPaths(conf, inputFile);
     FileOutputFormat.setOutputPath(conf, tempOut);

     // Set job specific distributed cache file (query file)
     DistributedCache.addCacheFile(topicFile.toUri(), conf);

     return conf ;
   } 

   public static void main(String[] args) throws Exception {
     Path tempOut = new Path(tempName);
     Path tempIn = new Path(tempName + "/part-00000");
     Path inputFile = new Path(args[0]);
     Path topicFile = new Path(args[1]);
     Path topicNewFile = new Path(args[2]);
     java.util.Map<String, Long> queryCounts = new HashMap<String, Long>();
		
     // Stop if out file exists
     FileSystem hdfs = FileSystem.get(new Configuration());
     if (hdfs.exists(topicNewFile)) {
       System.err.println("Output file " + topicNewFile + " already exists.");
       System.exit(1); 
     }
     hdfs.delete(tempOut, true);

     // Run the job
     JobConf conf = configureJob("QueryTermCount", inputFile, tempOut, topicFile);
     JobClient.runJob(conf);
 
     // Get temporary file for global statistics
     try {
       String tempLine;
       FSDataInputStream dis = hdfs.open(tempIn); 
       while ((tempLine = dis.readLine()) != null) { // deprecated, but it works
         String[] fields = tempLine.split("\t");
         queryCounts.put(fields[0], new Long(fields[1]));
       }
       dis.close();
     } catch (Exception e) {
       System.err.println(StringUtils.stringifyException(e));
       System.exit(1);
     }

     // Write new topic file with global statistics
     try {
       String tempLine;
       FSDataOutputStream dos = hdfs.create(topicNewFile);
       dos.writeChars("#MIREX-COMMENT: query term weight, document frequency, collection frequency (for each term)\n"); 
       dos.writeChars("#MIREX-COLLECTION:" + inputFile + "\n");
       dos.writeChars("#" + CollectionLength + ":" + queryCounts.get(CollectionLength) + "\n");
       dos.writeChars("#" + NumberOfDocs + ":" + queryCounts.get(NumberOfDocs) + "\n");

       FSDataInputStream dis = hdfs.open(topicFile);
       while ((tempLine = dis.readLine()) != null) {
         String [] fields = tempLine.split(":");
         dos.writeChars(fields[0] + ":");
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
           dos.writeChars(terms[i] + "=1=" + df.toString() + "=" + cf.toString());
           if (i < terms.length - 1) dos.writeChars(" "); 
         }
         dos.writeChars("\n");
      }
      dis.close();
      dos.close();
    } catch (Exception e) {
      System.err.println(StringUtils.stringifyException(e));
      System.exit(1);
    }
    hdfs.close();
  }

}
