
HADOOP_CMD="/usr/local/soft/hadoop/bin/hadoop"
STREAM_JAR_PATH="/usr/local/soft/hadoop/share/hadoop/tools/lib/hadoop-streaming-2.7.3.jar"

#INPUT_FILE_PATH_1="/part-00151"
INPUT_FILE_PATH_1="/music_uis.data"
OUTPUT_PATH_1="/output1"
OUTPUT_PATH_2="/output2"
OUTPUT_PATH_3="/output3"

$HADOOP_CMD fs -rmr -skipTrash $OUTPUT_PATH_1
$HADOOP_CMD fs -rmr -skipTrash $OUTPUT_PATH_2
$HADOOP_CMD fs -rmr -skipTrash $OUTPUT_PATH_3

# Step 1.
$HADOOP_CMD jar $STREAM_JAR_PATH \
		-input $INPUT_FILE_PATH_1 \
		-output $OUTPUT_PATH_1 \
		-mapper "python 1_gen_ui_map.py" \
		-reducer "python 1_gen_ui_reduce.py" \
		-file ./1_gen_ui_map.py \
		-file ./1_gen_ui_reduce.py

# Step 2.
$HADOOP_CMD jar $STREAM_JAR_PATH \
		-input $OUTPUT_PATH_1 \
		-output $OUTPUT_PATH_2 \
		-mapper "python 2_gen_ii_pair_map.py" \
		-reducer "python 2_gen_ii_pair_reduce.py" \
		-file ./2_gen_ii_pair_map.py \
		-file ./2_gen_ii_pair_reduce.py

# Step 3.
$HADOOP_CMD jar $STREAM_JAR_PATH \
		-input $OUTPUT_PATH_2 \
		-output $OUTPUT_PATH_3 \
		-mapper "python 3_sum_map.py" \
		-reducer "python 3_sum_reduce.py" \
		-file ./3_sum_map.py \
		-file ./3_sum_reduce.py

