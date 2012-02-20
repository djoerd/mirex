package nl.utwente.mirex.util;

import java.io.IOException;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.LineRecordReader;

public class KeyValueInputFormat extends FileInputFormat<Text, Text>{
	
	public KeyValueInputFormat() {		
	}
	
	protected boolean isSplitable(JobContext context, Path file) {
		return false;
	}
	
	@Override
	public RecordReader<Text, Text> createRecordReader(InputSplit split,
			TaskAttemptContext context) throws IOException, InterruptedException {
		LineRecordReader lrr = new LineRecordReader();		
		return new KeyValueRecordReader(lrr);
	}

	private class KeyValueRecordReader extends RecordReader<Text,Text> {
		private org.apache.commons.logging.Log Log = org.apache.commons.logging.LogFactory.getLog("org.apache.hadoop.mapred.MapTask"); 
		private RecordReader<LongWritable, Text> reader;
		
		private LongWritable position = new LongWritable();
		private Text key = new Text();
		private Text value = new Text();
		
		public KeyValueRecordReader(RecordReader<LongWritable, Text> reader) {
			if (reader==null) Log.error("Test");
			this.reader = reader;
		}
		
		@Override
		public void close() throws IOException {
			reader.close();
		}

		@Override
		public Text getCurrentKey() throws IOException, InterruptedException {
			return key;
		}

		@Override
		public Text getCurrentValue() throws IOException, InterruptedException {
			return value;
		}

		@Override
		public float getProgress() throws IOException, InterruptedException {
			return reader.getProgress();
		}

		@Override
		public void initialize(InputSplit arg0, TaskAttemptContext arg1)
				throws IOException, InterruptedException {
			reader.initialize(arg0, arg1);
		}

		@Override
		public boolean nextKeyValue() throws IOException, InterruptedException {
			boolean hasNext = reader.nextKeyValue();
			if (hasNext) {
				this.position.set(reader.getCurrentKey().get());
				String value = reader.getCurrentValue().toString();
				String[] fields = value.split("\t", 2);
				this.key.set(fields[0]);
				this.value.set(fields[1]);
				//Log.info("Read "+this.value.toString());
			}
			return hasNext;
		}
		
	}
}
