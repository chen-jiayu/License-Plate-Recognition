# 資料集，可根據需要增加英文或其它字元
DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# 分類數量
num_classes = len(DIGITS) + 1     # 資料集字元數+特殊識別符號

# 圖片大小，32 x 256
OUTPUT_SHAPE = (32, 256)

# 學習率
INITIAL_LEARNING_RATE = 1e-3
DECAY_STEPS = 5000
REPORT_STEPS = 100
LEARNING_RATE_DECAY_FACTOR = 0.9
MOMENTUM = 0.9

# LSTM網路層次
num_hidden = 128
num_layers = 2

# 訓練輪次、批量大小
num_epochs = 50000
BATCHES = 10
BATCH_SIZE = 32
TRAIN_SIZE = BATCHES * BATCH_SIZE

# 資料集目錄、模型目錄
data_dir = '/tmp/lstm_ctc_data/'
model_dir = '/tmp/lstm_ctc_model/'

# 生成椒鹽噪聲
def img_salt_pepper_noise(src,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        if random.randint(0,1)==0:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=255
    return NoiseImg

# 隨機生成不定長圖片集
def gen_text(cnt):
    # 設定文字字型和大小
    font_path = '/data/work/tensorflow/fonts/arial.ttf'
    font_size = 30
    font=ImageFont.truetype(font_path,font_size)

for i in range(cnt):
        # 隨機生成1到10位的不定長數字
        rnd = random.randint(1, 10)
        text = ''
        for j in range(rnd):
            text = text + DIGITS[random.randint(0, len(DIGITS) - 1)]

# 生成圖片並繪上文字
        img=Image.new("RGB",(256,32))
        draw=ImageDraw.Draw(img)
        draw.text((1,1),text,font=font,fill='white')
        img=np.array(img)

# 隨機疊加椒鹽噪聲並儲存影象
        img = img_salt_pepper_noise(img, float(random.randint(1,10)/100.0))
        cv2.imwrite(data_dir + text + '_' + str(i+1) + '.jpg',img)

# 序列轉為稀疏矩陣
# 輸入：序列
# 輸出：indices非零座標點，values資料值，shape稀疏矩陣大小
def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []

for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

return indices, values, shape

# 稀疏矩陣轉為序列
# 輸入：稀疏矩陣
# 輸出：序列
def decode_sparse_tensor(sparse_tensor):
    decoded_indexes = list()
    current_i = 0
    current_seq = []

for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)

result = []
    for index in decoded_indexes:
        result.append(decode_a_seq(index, sparse_tensor))
    return result

# 序列編碼轉換
def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        str = DIGITS[spars_tensor[1][m]]
        decoded.append(str)
    return decoded

# 將檔案和標籤讀到記憶體，減少磁碟IO
def get_file_text_array():
    file_name_array=[]
    text_array=[]

for parent, dirnames, filenames in os.walk(data_dir):
        file_name_array=filenames

for f in file_name_array:
        text = f.split('_')[0]
        text_array.append(text)

return file_name_array,text_array

# 獲取訓練的批量資料
def get_next_batch(file_name_array,text_array,batch_size=64):
    inputs = np.zeros([batch_size, OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]])
    codes = []

# 獲取訓練樣本
    for i in range(batch_size):
        index = random.randint(0, len(file_name_array) - 1)
        image = cv2.imread(data_dir + file_name_array[index])
        image = cv2.resize(image, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]), 3)
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        text = text_array[index]

# 矩陣轉置
        inputs[i, :] = np.transpose(image.reshape((OUTPUT_SHAPE[0], OUTPUT_SHAPE[1])))
        # 標籤轉成列表
        codes.append(list(text))

# 標籤轉成稀疏矩陣
    targets = [np.asarray(i) for i in codes]
    sparse_targets = sparse_tuple_from(targets)
    seq_len = np.ones(inputs.shape[0]) * OUTPUT_SHAPE[1]

return inputs, sparse_targets, seq_len

def get_train_model():
    # 輸入
    inputs = tf.placeholder(tf.float32, [None, None, OUTPUT_SHAPE[0]]) 

# 稀疏矩陣
    targets = tf.sparse_placeholder(tf.int32)

# 序列長度 [batch_size,]
    seq_len = tf.placeholder(tf.int32, [None])

# 定義LSTM網路
    cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)      # old
    outputs, _ = tf.nn.dynamic_rnn(cell, inputs, seq_len, dtype=tf.float32)
    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]

outputs = tf.reshape(outputs, [-1, num_hidden])
    W = tf.Variable(tf.truncated_normal([num_hidden,
                                         num_classes],
                                        stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0., shape=[num_classes]), name="b")
    logits = tf.matmul(outputs, W) + b
    logits = tf.reshape(logits, [batch_s, -1, num_classes])

# 轉置矩陣
    logits = tf.transpose(logits, (1, 0, 2))

return logits, inputs, targets, seq_len, W, b 

# 準確性評估
# 輸入：預測結果序列 decoded_list ,目標序列 test_targets
# 返回：準確率
def report_accuracy(decoded_list, test_targets):
    original_list = decode_sparse_tensor(test_targets)
    detected_list = decode_sparse_tensor(decoded_list)

# 正確數量
    true_numer = 0

# 預測序列與目標序列的維度不一致，說明有些預測失敗，直接返回
    if len(original_list) != len(detected_list):
        print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
              " test and detect length desn't match")
        return

# 比較預測序列與結果序列是否一致，並統計準確率        
    print("T/F: original(length) <-------> detectcted(length)")
    for idx, number in enumerate(original_list):
        detect_number = detected_list[idx]
        hit = (number == detect_number)
        print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
        if hit:
            true_numer = true_numer + 1
    accuracy = true_numer * 1.0 / len(original_list)
    print("Test Accuracy:", accuracy)

return accuracy


def train():
    # 獲取訓練樣本資料
    file_name_array, text_array = get_file_text_array()

# 定義學習率
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                               global_step,
                                               DECAY_STEPS,
                                               LEARNING_RATE_DECAY_FACTOR,
                                               staircase=True)
    # 獲取網路結構
    logits, inputs, targets, seq_len, W, b = get_train_model()

# 設定損失函式
    loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len)
    cost = tf.reduce_mean(loss)

# 設定優化器
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

with tf.Session() as session:
        session.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

for curr_epoch in range(num_epochs):
            train_cost = 0
            train_ler = 0
            for batch in range(BATCHES):
                # 訓練模型
                train_inputs, train_targets, train_seq_len = get_next_batch(file_name_array, text_array, BATCH_SIZE)
                feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}
                b_loss, b_targets, b_logits, b_seq_len, b_cost, steps, _ = session.run(
                    [loss, targets, logits, seq_len, cost, global_step, optimizer], feed)

# 評估模型
                if steps > 0 and steps % REPORT_STEPS == 0:
                    test_inputs, test_targets, test_seq_len = get_next_batch(file_name_array, text_array, BATCH_SIZE)
                    test_feed = {inputs: test_inputs,targets: test_targets,seq_len: test_seq_len}
                    dd, log_probs, accuracy = session.run([decoded[0], log_prob, acc], test_feed)
                    report_accuracy(dd, test_targets)

# 儲存識別模型
                    save_path = saver.save(session, model_dir + "lstm_ctc_model.ctpk",global_step=steps)

c = b_cost
                train_cost += c * BATCH_SIZE

train_cost /= TRAIN_SIZE
            # 計算 loss
            train_inputs, train_targets, train_seq_len = get_next_batch(file_name_array, text_array, BATCH_SIZE)
            val_feed = {inputs: train_inputs,targets: train_targets,seq_len: train_seq_len}
            val_cost, val_ler, lr, steps = session.run([cost, acc, learning_rate, global_step], feed_dict=val_feed)

log = "{} Epoch {}/{}, steps = {}, train_cost = {:.3f}, val_cost = {:.3f}"
            print(log.format(curr_epoch + 1, num_epochs, steps, train_cost, val_cost))


# LSTM+CTC 文字識別能力封裝
# 輸入：圖片
# 輸出：識別結果文字
def predict(image):

# 獲取網路結構
    logits, inputs, targets, seq_len, W, b = get_train_model()
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

saver = tf.train.Saver()
    with tf.Session() as sess:
        # 載入模型
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        # 影象預處理
        image = cv2.resize(image, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]), 3)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        pred_inputs = np.zeros([1, OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]])
        pred_inputs[0, :] = np.transpose(image.reshape((OUTPUT_SHAPE[0], OUTPUT_SHAPE[1])))
        pred_seq_len = np.ones(1) * OUTPUT_SHAPE[1]
        # 模型預測
        pred_feed = {inputs: pred_inputs,seq_len: pred_seq_len}
        dd, log_probs = sess.run([decoded[0], log_prob], pred_feed)
        # 識別結果轉換
        detected_list = decode_sparse_tensor(dd)[0]
        detected_text = ''
        for d in detected_list:
            detected_text = detected_text + d

return detected_text