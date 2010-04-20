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

import org.apache.hadoop.filecache.DistributedCache;
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
   private static final String tempFile = SysName + "-tmp";

   public static class Map extends MapReduceBase implements Mapper<Text, Text, Text, LongWritable> {

     private static final String TOKENIZER = "[^0-9A-Za-z]+";
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


   public static void main(String[] args) throws Exception {
     // Set job configuration
     JobConf conf = new JobConf(TrecRun.class);
     conf.setJobName("QueryTermCount");
		
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
     FileInputFormat.setInputPaths(conf, new Path(args[0]));
     FileOutputFormat.setOutputPath(conf, new Path(tempFile));
		
     // Set job specific distributed cache file (query file)
     DistributedCache.addCacheFile(new Path(args[1]).toUri(), conf);
	
     // Run the job
     JobClient.runJob(conf);
 
       // for each query, score the document
 /*      if (doclength > 0) {
         Iterator iterator = trecQueries.keySet().iterator();
         while (iterator.hasNext()) {
           String qid = (String) iterator.next();
           String [] qterms = (String []) trecQueries.get(qid);
           Double score = scoreDocumentLM(qterms, docTF, doclength);
           if (score != 0.0d) 
             output.collect(new Text(qid), new Text(key.toString() + "\t" + score.toString()));
         }
       }
*/

  }

}
