package nl.utwente.mirex.util;

import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;

import edu.cmu.lemurproject.WarcRecord;
import edu.cmu.lemurproject.WarcFileInputFormat;
import edu.cmu.lemurproject.WritableWarcRecord;

public class WarcTextConverterInputFormat extends FileInputFormat<Text, Text>{

	private WarcFileInputFormat fileinput;
	
	private final static Pattern
                headerPat = Pattern.compile("$(.*?)<",  
			Pattern.CASE_INSENSITIVE | Pattern.MULTILINE | Pattern.DOTALL),
		scriptPat = Pattern.compile("(?s)<script(.*?)</script>", 
			Pattern.CASE_INSENSITIVE | Pattern.MULTILINE | Pattern.DOTALL),
		tagsPat = Pattern.compile("<[^>]+>", Pattern.CASE_INSENSITIVE | Pattern.MULTILINE | Pattern.DOTALL),
		spacePat = Pattern.compile("[ \n\r\t]+", Pattern.CASE_INSENSITIVE | Pattern.MULTILINE | Pattern.DOTALL),
		noIndexPat = Pattern.compile("/index\\.[a-z][a-z][a-z][a-z]?$", Pattern.CASE_INSENSITIVE);

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
			String webpage, url;
			
			if (hasNext) {
				WarcRecord thisRecord = reader.getCurrentValue().getRecord();
				this.key.set(thisRecord.getHeaderMetadataItem("WARC-TREC-ID"));
				url = thisRecord.getHeaderMetadataItem("WARC-Target-URI");
                                url = noIndexPat.matcher(url).replaceFirst("/");
				webpage = thisRecord.getContentUTF8();
				webpage = headerPat.matcher(webpage).replaceFirst("<");
				webpage = scriptPat.matcher(webpage).replaceAll(" ");
				webpage = tagsPat.matcher(webpage).replaceAll(" ");
				webpage = spacePat.matcher(webpage).replaceAll(" ");
				this.value.set(url + "\t" + webpage);
			}
			return hasNext;
		}
		
	}
}
