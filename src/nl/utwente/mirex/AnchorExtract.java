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
 *            Guido van der Zanden
 * 
 * About:
 * ------
 * 
 * Extract anchor text from HTML documents. 
 *  Input: ClueWeb warc files (gzipped)
 *  Output: (WARC-TREC-ID, anchor texts), tab separated (gzipped)
 * 
 * Only for documents inside the collection.
 * Documents in the collection without inlinks are not listed.
 * 
 * Djoerd Hiemstra, February 2009
 */

package nl.utwente.mirex;

import java.io.IOException;
import java.util.Iterator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextOutputFormat;

import edu.cmu.lemurproject.WarcRecord;
import edu.cmu.lemurproject.WritableWarcRecord;
import edu.cmu.lemurproject.WarcFileInputFormat;

public class AnchorExtract {

   private final static String MirexId = "MIREX-TREC-ID: ";
   private final static Pattern mirexIdPat = Pattern.compile(MirexId + "(.+)$");
   private final static int maxCapacity = 10000000; // not much more than 10 MB anchors

   public static class Map extends MapReduceBase implements Mapper<LongWritable, WritableWarcRecord, Text, Text> {

     private final static Pattern
       anchorPat = Pattern.compile("(?s)<a ([^>]*)href=[\"']?([^> '\"]+)([^>]*)>", Pattern.CASE_INSENSITIVE),
       relUrlPat = Pattern.compile("^/"),
       absUrlPat = Pattern.compile("^[a-z]+://"),
       nofollowPat = Pattern.compile("rel=[\"']?nofollow", Pattern.CASE_INSENSITIVE); // ignore links with rel="nofollow"
     private final static String noIndexHTML = "/index\\.[a-z][a-z][a-z][a-z]?$";

     private static String makeAbsoluteUrl(String targetUrl, String relativeUrl) {
       /* takes url of web page (targetUrl) and relative url to make absolute url */
       // assert !targetUrl.equals("");
       String absUrl;
       targetUrl = absUrlPat.matcher(targetUrl).replaceAll(""); // remove protocol header
       relativeUrl = relativeUrl.replaceAll("[ \n\r\t]","");
       Matcher matcher = relUrlPat.matcher(relativeUrl);
       if (matcher.find())
         absUrl = targetUrl.replaceAll("/.*$", "") + relativeUrl;
       else {
         matcher = absUrlPat.matcher(relativeUrl);
         if (matcher.find()) absUrl = matcher.replaceAll("");
         else absUrl = targetUrl.replaceAll("/[^/]+$", "") + '/' + relativeUrl;
       }
       return "http://" + absUrl.replaceAll("/.[^/]+/\\.\\./|//", "/").replaceFirst(noIndexHTML, "/");
     }

     public void map(LongWritable key, WritableWarcRecord value, OutputCollector<Text, Text> output, Reporter reporter) throws IOException {
       String baseUri, trecId, content;
       Text link = new Text(), anchor = new Text();
       Matcher matcher;
       WarcRecord thisRecord = value.getRecord();
       if (thisRecord.getHeaderRecordType().equals("response")) {
         baseUri = thisRecord.getHeaderMetadataItem("WARC-Target-URI").replaceFirst(noIndexHTML, "/");
         trecId = thisRecord.getHeaderMetadataItem("WARC-TREC-ID");
         link.set(baseUri);
         anchor.set(MirexId + trecId);
         output.collect(link, anchor);           // we want to keep track of the TREC-IDs
         content = thisRecord.getContentUTF8();
         matcher = anchorPat.matcher(content);
         while(matcher.find()) {
           Matcher nomatch = nofollowPat.matcher(matcher.group(1) + matcher.group(3));
           if (!nomatch.find()) {
             link.set(makeAbsoluteUrl(baseUri, matcher.group(2)));
             anchor.set(matcher.group(4).replaceAll("<[^>]+>|[ \n\t\r]+", " "));
             output.collect(link, anchor);
           }
         }
       }
     }
   }

   public static class Combine extends MapReduceBase implements Reducer<Text, Text, Text, Text> {

     public void reduce(Text key, Iterator<Text> values, OutputCollector<Text, Text> output, Reporter reporter) throws IOException {
       boolean first = true;
       String trecId = "";
       StringBuilder anchors = new StringBuilder();
       while (values.hasNext()) {
         String anchor = values.next().toString();
         Matcher matcher = mirexIdPat.matcher(anchor);
         if (matcher.find()) {
           output.collect(key, new Text(anchor));
         }
         else {
           if (anchors.length() < maxCapacity) {
             if (first) { anchors.append(anchor); first = false; }
             else { anchors.append("\t").append(anchor); }
           }
         }
       }
       if (!first) {
         output.collect(key, new Text(anchors.toString()));
       }
     }

   }

   public static class Reduce extends MapReduceBase implements Reducer<Text, Text, Text, Text> {

     public void reduce(Text key, Iterator<Text> values, OutputCollector<Text, Text> output, Reporter reporter) throws IOException {

       boolean found = false;
       String trecId = "";
       StringBuilder anchors = new StringBuilder();
       anchors.append(key.toString());
      
       while (values.hasNext()) {
         String anchor = values.next().toString();
         Matcher matcher = mirexIdPat.matcher(anchor);
         if (matcher.find()) {
           trecId = matcher.group(1);
         }
         else if (anchors.length() < maxCapacity) {
           anchors.append("\t").append(anchor);
           found = true;
         } 
       }
       if (found && trecId != "") {
         output.collect(new Text(trecId), new Text(anchors.toString()));
         if (anchors.length() >= maxCapacity) { 
           System.err.println("Warning: Maximum capacity reached for: " + trecId);
         }
       }
     }
   }


   public static void main(String[] args) throws Exception {
     JobConf conf = new JobConf(AnchorExtract.class);

     conf.setJobName("anchorextract");

     conf.setMapperClass(Map.class);
     conf.setMapOutputKeyClass(Text.class);
     conf.setMapOutputValueClass(Text.class);

     conf.setCombinerClass(Combine.class);

     conf.setReducerClass(Reduce.class);
     conf.setOutputKeyClass(Text.class);
     conf.setOutputValueClass(Text.class);

     conf.setInputFormat(WarcFileInputFormat.class);
     conf.setOutputFormat(TextOutputFormat.class);

     FileInputFormat.setInputPaths(conf, new Path(args[0]));
     FileOutputFormat.setOutputPath(conf, new Path(args[1]));
     FileOutputFormat.setCompressOutput(conf, true);
     FileOutputFormat.setOutputCompressorClass(conf, GzipCodec.class);

     JobClient client = new JobClient();
     client.setConf(conf);
     client.runJob(conf); 
   }
}

