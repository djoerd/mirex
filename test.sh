#!/bin/bash
hadoop fs -mkdir warc-files
hadoop fs -put data/test.warc.gz ./warc-files/
hadoop fs -put data/wt2010-topics.queries-only ./
hadoop jar mirex-0.3.jar nl.utwente.mirex.AnchorExtract warc-files/* anchors
hadoop jar mirex-0.3.jar nl.utwente.mirex.TrecRun          KEYVAL anchors/* TrecOut wt2010-topics.queries-only
hadoop fs -cat /user/alyr/TrecOut/part*
hadoop jar mirex-0.3.jar nl.utwente.mirex.QueryTermCount   KEYVAL anchors/*  wt2010-topics.queries-only  wt2010-topics.stats
hadoop fs -cat /user/alyr/wt2010-topics.stats
hadoop jar mirex-0.3.jar nl.utwente.mirex.TrecRunBaselines KEYVAL anchors/* BaselineOut wt2010-topics.stats
hadoop fs -cat /user/alyr/BaselineOut/part*
#
# Warc files
#
hadoop jar mirex-0.3.jar nl.utwente.mirex.TrecRun          WARC warc-files/* TrecOut2 wt2010-topics.queries-only
hadoop fs -cat /user/alyr/TrecOut2/part*
hadoop jar mirex-0.3.jar nl.utwente.mirex.QueryTermCount   WARC warc-files/* wt2010-topics.queries-only  wt2010-topics.stats2
hadoop fs -cat /user/alyr/wt2010-topics.stats2
hadoop jar mirex-0.3.jar nl.utwente.mirex.TrecRunBaselines WARC warc-files/* BaselineOut2 wt2010-topics.stats2
hadoop fs -cat /user/alyr/BaselineOut2/part*