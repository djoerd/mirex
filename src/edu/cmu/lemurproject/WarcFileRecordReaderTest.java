package edu.cmu.lemurproject;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.TaskAttemptID;
import org.apache.hadoop.mapreduce.TaskID;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

/**
 * Quick and dirty test program 
 * @author alyr
 *
 */
public class WarcFileRecordReaderTest {
	public static void main(String[] args) throws IOException, InterruptedException {
		Job job = new Job();
		TaskAttemptContext context = new TaskAttemptContext(job.getConfiguration(), new TaskAttemptID(new TaskID(), 1));
		String fileName = "FILL-IN-TEST-FILE";		
		WarcFileInputFormat wi = new WarcFileInputFormat();		
		WarcFileInputFormat.addInputPath(job, new Path(fileName));
		List<InputSplit> splits = wi.getSplits(job);
		System.out.println(splits.size());
		for (InputSplit split: splits) { 	
			System.out.println(((FileSplit)split).getStart());
		}
		System.exit(1);
		for (InputSplit split: splits) { 		
			WarcFileRecordReader wr = new WarcFileRecordReader();
			wr.initialize(split, context);
			while (wr.nextKeyValue()) {
				System.out.println(wr.getCurrentKey().toString());
			}
		}
	}
}
