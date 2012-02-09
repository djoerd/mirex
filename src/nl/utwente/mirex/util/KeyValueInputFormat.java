package nl.utwente.mirex.util;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;

public class KeyValueInputFormat extends FileInputFormat<Text, Text>{
	private TextInputFormat fileinput;
	
	public KeyValueInputFormat() {
		fileinput = new TextInputFormat();
	}
	
	@Override
	public RecordReader<Text, Text> createRecordReader(InputSplit split,
			TaskAttemptContext context) throws IOException, InterruptedException {
		RecordReader<LongWritable, Text> reader = fileinput.createRecordReader(split, context);
		return new KeyValueRecordReader(reader);
	}

	private class KeyValueRecordReader extends RecordReader<Text,Text> {

		private RecordReader<LongWritable, Text> reader;
		
		private LongWritable position;
		private Text key = new Text();
		private Text value = new Text();
		
		public KeyValueRecordReader(RecordReader<LongWritable, Text> reader) {
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
			}
			return hasNext;
		}
		
	}
}