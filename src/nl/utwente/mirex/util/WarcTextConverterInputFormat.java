package nl.utwente.mirex.util;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;

import edu.cmu.lemurproject.WarcFileInputFormat;
import edu.cmu.lemurproject.WritableWarcRecord;

public class WarcTextConverterInputFormat extends FileInputFormat<Text, Text>{
	private WarcFileInputFormat fileinput;
	
	public WarcTextConverterInputFormat() {
		fileinput = new WarcFileInputFormat();
	}
	
	@Override
	public RecordReader<Text, Text> createRecordReader(InputSplit split,
			TaskAttemptContext context) throws IOException, InterruptedException {
		RecordReader<LongWritable, WritableWarcRecord> reader = fileinput.createRecordReader(split, context);
		return new WarcTextConverterRecordReader(reader);
	}

	private class WarcTextConverterRecordReader extends RecordReader<Text,Text> {

		private RecordReader<LongWritable, WritableWarcRecord> reader;
		
		private Text key = new Text();
		private Text value = new Text();
		
		public WarcTextConverterRecordReader(RecordReader<LongWritable, WritableWarcRecord> reader) {
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
				this.key.set(reader.getCurrentValue().getRecord().getHeaderMetadataItem("WARC-TREC-ID"));
				this.value.set(reader.getCurrentValue().getRecord().getContentUTF8());
			}
			return hasNext;
		}
		
	}
}
