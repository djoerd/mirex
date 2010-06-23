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

/**
 * <b>Runs MapReduce job:</b> Runs several baseline experiments. 
 * Do several TREC baseline runs, such as linear interpolation smoothing
 * (with parameter sweep), Dirichlet smoothing, BM25.
 *  Input: (argument 1) Document representation (WARC-TREC-ID, text), 
 *         tab separated
 *         (argument 3) TREC ClueWeb queries with global statistic from
 *         QueryTermCount.java (TREC-QUERY-ID, Query terms+frequencies), 
 *         separated by a colon (":")
 *  Output: (argument 2) (TREC-QUERY-ID ":" Model, WARC-TREC-ID, score), tab separated 
 * @author Djoerd Hiemstra
 * @since 0.2
 * @see QueryTermCount
 */
public class TrecRunBaselines {

   /**
    * -- Mapper: Runs all queries and all models on one document. 
    */
   public static class Map extends MapReduceBase implements Mapper<Text, Text, Text, Text> {

     private static final String TOKENIZER = "[^0-9A-Za-z]+";
     private java.util.Map<String, String[]> trecQueries = new HashMap<String, String[]>();
     private java.util.Map<String, Integer> queryTerms = new HashMap<String, Integer>();
     private Long globalNDocs = 0l;
     private Long globalCollLength = 0l;
     private Double globalAvgDocLength = 0.0d;

     private class TermInfo {
       public final String term;
       public final Integer qtf;
       public final Long df;
       public final Long cf;

       public TermInfo(String fileRep) {
         String [] parts = fileRep.split("=");
         if (parts.length != 4) throw new ArrayIndexOutOfBoundsException("Error in query format");
         this.term = parts[0];
         this.qtf = Integer.parseInt(parts[1]);
         this.df = Long.parseLong(parts[2]);
         this.cf = Long.parseLong(parts[3]);
       }
     }

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

       while ((queryString = fis.readLine()) != null && queryString.startsWith("#MIREX")) {
         if (queryString.startsWith("#MIREX-LENGTH:")) {
           String [] parts = queryString.split(":");
           globalCollLength = Long.parseLong(parts[1]);
         } 
         if (queryString.startsWith("#MIREX-NDOCS:")) {
           String [] parts = queryString.split(":");
           globalNDocs = Long.parseLong(parts[1]);
         }
       }
       if (queryString == null) throw new IOException("Wrong format, no queries");
       if (globalNDocs == 0 || globalCollLength == 0) throw new IOException("Wrong format, use results of QueryTermCount.java");
       globalAvgDocLength = globalCollLength.doubleValue() / globalNDocs.doubleValue();
       while (queryString != null) {
         queryString = queryString.toLowerCase();
         String [] fields = queryString.split(":");
         String [] terms = fields[1].split(" ");
         trecQueries.put(fields[0], terms);
         for (int i=0; i < terms.length; i++) {
           TermInfo termInfo = new TermInfo(terms[i]);
           queryTerms.put(termInfo.term, 1);
         }
         queryString = fis.readLine();
       } 
     }


    // ALL RETRIEVAL MODELS GO BELOW -- These would normally be private methods, but this way they end up in javadoc automatically
    
    /**
     * Computes score using a language model with NO smoothing and a document length prior (Original model, see TrecRun.java)
     * @param qterms array of strings
     * @param docTF document term frequencies
     * @param doclength document length
     * @see TrecRun
     */
     public Double scoreDocumentLMno(String[] qterms, java.util.Map<String, Integer> docTF, Long doclength) {
       Double score = 1.0d;
       for (int i=0; i < qterms.length; i++) {
         TermInfo termInfo = new TermInfo(qterms[i]);
         Integer tf = (Integer) docTF.get(termInfo.term);
         if (tf != null) {
           for (int j=0; j < termInfo.qtf; j++) score *= (new Double(tf) / new Double(doclength));
         }
         else return 0.0d;  // no match
       }
       return score * doclength; // length prior
     }

    /**
     * Computes score using a language model with linear interpolation smoothing and a document length prior
     * @param qterms array of strings
     * @param docTF document term frequencies
     * @param doclength document length
     * @param lambda parameter lambda (0 < lambda < 1)
     */
     public Double scoreDocumentLMs(String[] qterms, java.util.Map<String, Integer> docTF, Long doclength, Double lambda) {
       Double score = 0.0d;
       for (int i=0; i < qterms.length; i++) {
         TermInfo termInfo = new TermInfo(qterms[i]);
         Integer tf = (Integer) docTF.get(termInfo.term);
         if (tf != null) {
           score += termInfo.qtf * Math.log(1 + (tf.doubleValue() * globalCollLength.doubleValue() * lambda) 
                                               / (termInfo.cf.doubleValue() * doclength.doubleValue() * (1 - lambda)) );
         }
       }
       if (score > 0.0d) return score + Math.log(doclength); // length prior
       else return 0.0d;
     }

    /**
     * Computes score using Okapi's BM25
     * @param qterms array of strings
     * @param docTF document term frequencies
     * @param doclength document length
     * @param k1 parameter k1 (k1 > 0)
     * @param b parameter b (0 < b < 1)
     */
     public Double scoreDocumentBM25(String[] qterms, java.util.Map<String, Integer> docTF, Long doclength, Double k1, Double b) {
       Double score = 0.0d;
       for (int i=0; i < qterms.length; i++) {
         TermInfo termInfo = new TermInfo(qterms[i]);
         Integer tf = (Integer) docTF.get(termInfo.term);
         if (tf != null) {
             Double K = k1 * ((1 - b) + b*(doclength.doubleValue() / globalAvgDocLength)); 
             score += termInfo.qtf * (((k1 + 1) * tf ) / (K + tf)) * Math.log((globalNDocs - termInfo.df + 0.5) / (termInfo.df + 0.5));
         }
       }
       return score;
     }

    /**
     * Computes score using a language model with Dirichlet smoothing
     * @param qterms array of strings
     * @param docTF document term frequencies
     * @param doclength document length
     * @param mu smoothing parameter mu (mu > 0)
     */
     public Double scoreDocumentLMdi(String[] qterms, java.util.Map<String, Integer> docTF, Long doclength, Double mu) {
       /* Language model with Dirichlet smoothing */
       Double score = 0.0d;
       for (int i=0; i < qterms.length; i++) {
         TermInfo termInfo = new TermInfo(qterms[i]);
         Integer tf = (Integer) docTF.get(termInfo.term);
         if (tf == null) tf = 0;
         score += termInfo.qtf * (Math.log(tf * (globalCollLength / termInfo.cf) + mu) - Math.log(doclength + mu)); 
       }
       if (score > 0) return score; else return 0.0d; // a matching document might get score smaller than zero in theory
     }

     // ALL RETRIEVAL MODELS GO ABOVE 


     /**
      * @param key TREC-ID
      * @param value document text
      * @param output (Query-ID ":" Model, TREC-ID, score)
      */
     public void map(Text key, Text value, OutputCollector<Text, Text> output, Reporter reporter) throws IOException {

       // Store tf's of document only for term that is in one of the queries
       java.util.Map<String, Integer> docTF = new HashMap<String, Integer>();
       Long doclength = 0l; // one more, so at least 1.
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
       if (doclength > 0l) {
         Iterator iterator = trecQueries.keySet().iterator();
         while (iterator.hasNext()) {
           Double score;
           String qid = (String) iterator.next();
           String [] qterms = (String []) trecQueries.get(qid);

           if ((score = scoreDocumentLMno(qterms, docTF, doclength)) != 0.0d)
             output.collect(new Text(qid + ":LMno"), new Text(key.toString() + "\t" + score.toString()));
           
           // example of parameter sweep 
           for (Double lambda = 0.1d; lambda < 1.0d; lambda += 0.2d) 
             if ((score = scoreDocumentLMs(qterms, docTF, doclength, lambda)) != 0.0d)
               output.collect(new Text(qid + ":LMs" + String.format("%1.1f", lambda)), 
                              new Text(key.toString() + "\t" + score.toString()));

           if ((score = scoreDocumentLMdi(qterms, docTF, doclength, 2500d)) != 0.0d)
             output.collect(new Text(qid + ":LMdi"), new Text(key.toString() + "\t" + score.toString()));

           if ((score = scoreDocumentBM25(qterms, docTF, doclength, 1.2d, 0.75d)) != 0.0d)
             output.collect(new Text(qid + ":BM25"), new Text(key.toString() + "\t" + score.toString()));

         }
       }
     }
   }

  /**
   * -- Reducer: Sorts the retrieved documents and takes the top 1000.
   */
   public static class Reduce extends MapReduceBase implements Reducer<Text, Text, Text, Text> {

     private static final Integer TOP = 1000;

     /**
      * @param key Query-ID ":" Model
      * @param values (TREC-ID, score)
      * @param output (Query-ID ":" Model, TREC-ID, score)
      */
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


  /**
   * Runs the MapReduce job "trec baseline runs"
   * @param args 0: path to parsed document collection (use AnchorExtract); 1: (non-existing) path that will contain run resutls; 2: MIREX query file
   * @usage. 
   * <code> % hadoop jar mirex-0.2.jar nl.utwente.mirex.TrecRunBaselines /user/hadoop/ClueWeb09_Anchors/* /user/hadoop/BaselineOut /user/hadoop/wt09-topics-stats.txt </code> 
   */
   public static void main(String[] args) throws Exception {
     // Set job configuration
     JobConf conf = new JobConf(TrecRun.class);
     conf.setJobName("MirexBaselineRuns");
		
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
