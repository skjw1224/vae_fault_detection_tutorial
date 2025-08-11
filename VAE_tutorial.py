import sys

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer, Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
import seaborn as sns

def compute_KDE(data, alpha=95):
    """
    Kernel Density Estimation(KDE)를 사용하여 데이터의 상위 alpha 백분위수를 계산한다..
    이를 통해 이상치 감지의 기준점인 UCL(Upper Control Limit)을 설정할 수 있다.
    """
    data = data.values
    sigma = np.std(data)
    n = len(data)
    h = 1.06 * sigma * np.float_power(n, -0.2)

    KDE = KernelDensity(kernel='gaussian', bandwidth=h).fit(data)
    score = KDE.sample(100000)
    ucl = np.percentile(score, alpha)
    return ucl

class Sampling(Layer):
    """
        잠재 공간 샘플링을 위한 함수.
        VAE에서 샘플링하는 과정은 reparameterization trick을 사용하여 역전파가 가능하게 만들고, latent vector z를 생성한다.
        args: 인코더에서 생성된 평균(z_mean)과 로그 분산(z_log_var)을 입력으로 받는다
        """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE_LossLayer(Layer):
    """
    재구축 오차와 KL divergence 손실을 계산하는 사용자 정의 레이어
    """
    def __init__(self, input_dim, **kwargs):
        super(VAE_LossLayer, self).__init__(**kwargs)
        self.input_dim = input_dim  # 입력 차원 저장

    def call(self, inputs):
        input_layer, decoder_output, z_mean, z_log_var = inputs
        # 재구축 오차 계산
        reconstruction_loss = tf.reduce_mean(tf.square(input_layer - decoder_output)) * self.input_dim
        # KL divergence 손실 계산
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        # 전체 손실에 추가
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        self.add_loss(vae_loss)
        return decoder_output  # 이 레이어는 출력에 대해 아무것도 변경하지 않습니다.

def compute_T2(PC_data, S_inverse):              # Hoteling T2 계산
    """
    Hoteling T2 통계량을 계산한다. 이는 다변량 데이터의 이상치를 감지하는 데 사용한다.
    """
    T_T_tensor = tf.convert_to_tensor(PC_data, dtype=tf.float32)
    T2_values = []
    for i in range(PC_data.shape[0]):
        T_i_T = tf.slice(T_T_tensor, [i, 0], [1, -1])
        T_i = tf.transpose(T_i_T)
        T2 = tf.matmul(tf.matmul(T_i_T, S_inverse), T_i)
        T2_values.append(tf.squeeze(T2))
    return tf.stack(T2_values, axis=0)

def compute_covariance_and_inverse(PC_train):
    """
    주어진 데이터로부터 공분산 행렬과 그 역행렬을 계산한다. 이는 T2 통계량 계산에 사용된다.
    """
    matrix_tensor = tf.convert_to_tensor(PC_train, dtype=tf.float32)
    mean_tensor = tf.reduce_mean(matrix_tensor, axis=0)
    centered_matrix = matrix_tensor - mean_tensor
    S = tf.matmul(centered_matrix, centered_matrix, transpose_a=True) / tf.cast(tf.shape(matrix_tensor)[0], tf.float32)
    identity_matrix = tf.eye(S.shape[0])
    L_inv = tf.linalg.triangular_solve(tf.linalg.cholesky(S), identity_matrix, lower=True)
    S_inverse = tf.matmul(tf.transpose(L_inv),L_inv)
    return S, S_inverse


def build_vae(input_dim, encoding_dim,learning_rate=0.001):
    """
      VAE 모델을 구축하는 함수. 입력 데이터의 차원, 인코딩된 잠재 공간의 차원, 학습률을 지정한다
      VAE는 인코더와 디코더로 구성된다 인코더는 입력 데이터를 잠재 공간의 벡터로 압축하고, 디코더는 이를 다시 원래의 데이터 공간으로 복원한다
      """
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoder_layer1 = Dense(input_dim, activation=tf.nn.leaky_relu)(input_layer)
    encoder_layer2 = Dense(32, activation=tf.nn.leaky_relu)(encoder_layer1)
    encoder_layer3 = Dense(16, activation=tf.nn.leaky_relu)(encoder_layer2)
    z_mean = Dense(encoding_dim, activation=tf.nn.leaky_relu)(encoder_layer3)
    z_log_var = Dense(encoding_dim, activation=tf.nn.leaky_relu)(encoder_layer3)
    z = Sampling()([z_mean,z_log_var])

    # Decoder
    decoder_layer1 = Dense(16, activation=tf.nn.leaky_relu)(z)
    decoder_layer2 = Dense(32, activation=tf.nn.leaky_relu)(decoder_layer1)
    decoder_layer3 = Dense(input_dim, activation=tf.nn.leaky_relu)(decoder_layer2)
    decoder_layer4 = Dense(input_dim, activation='linear')(decoder_layer3)


    # VAE loss
    custom_vae_loss_layer = VAE_LossLayer(input_dim)([input_layer, decoder_layer4, z_mean, z_log_var])

    # VAE model
    VAE = Model(input_layer, custom_vae_loss_layer)

    # Encoder model
    encoder = Model(input_layer, z)

    # Decoder model
    decoder = Model(z, decoder_layer4)


    optimizer=tf.optimizers.Adam(learning_rate=learning_rate)
    VAE.compile(optimizer=optimizer)

    return VAE, encoder, decoder

def compute_F1(fault_data, no_fault_data, UCL):
    """
      모니터링 성능을 정량화하는 F1 Score를 계산하는 함수.
      정상데이터의 모니터링 지표, fault데이터의 모니터링 지표, UCL을 받아서 F1 score를 계산한다.

      """
    TP = np.sum(fault_data > UCL)
    FP = np.sum(no_fault_data > UCL)
    FN = len(fault_data) - TP

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    return TP , FP, F1


"""
데이터 전처리 단계
"""
#load train data
data_train = pd.read_csv('train_test.csv',index_col=0)


print(data_train)

#  전처리를 위한 Standard Sclaer 객체 생성
scaler = StandardScaler()

# 훈련 데이터에 대해 스케일러 학습
scaler.fit(data_train)

# 훈련 데이터와 테스트 데이터 스케일링
data_train = scaler.transform(data_train)



# 학습 데이터와 검증 데이터로 분할
x_train, x_val = train_test_split(data_train, test_size=0.2, random_state=58,shuffle=True)

x_total = data_train.astype('float32')
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')



"""
VAE모델 선언 및 모델 학습 단계
"""
# 1. VAE모델 구축
# input_dim차원의 데이터(변수의 개수)를 encoding_dim의 차원까지 압축했다가 복원하는 VAE모델 선언
VAE, VAE_encoder, VAE_decoder = build_vae(input_dim=x_train.shape[1],encoding_dim=8,learning_rate=0.001)


# 2. VAE모델 학습
# overfitting을 막기 위한 earlystopping 선언, validation데이터에 대해서 성능향상이 20epoch동안 없을경우 학습을 종료하고 최고 성능일때의 모델로 복원
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)
VAE.fit(x_train,
                epochs=100,
                batch_size=32,
                validation_data=x_val,
                callbacks=[early_stopping])


# 3. 학습된 VAE모델 저장
VAE.save('VAE.h5')      #VAE.load_weights('VAE.h5') #저장된 모델 불러오는 코드, 모델을 똑같이 선언해 준 후에 사용해야 하며 가중치만 업데이트 하는 방식


# 4. 학습된 VAE모델의 latent 분포 확인
latent = VAE_encoder.predict(x_total)
latent = pd.DataFrame(latent)
plt.figure(figsize=(20,20))
sns.pairplot(latent)
plt.show()


"""
모니터링 지표인 재구축 오차(Square Prediction Error)와 Hotelling t2의 관리한계선(Upper Control Limit;UCL) 설정 단계
"""
# 5. 재구축오차(SPE)를 사용한 모니터링

# 5.1 재구축오차(SPE)의 UCL(Upper Control Limit)계산
total_recon = tf.reduce_sum(tf.square(x_total-VAE.predict(x_total)),axis=1)  #train 데이터의 재구축 오차 계산
total_recon = pd.DataFrame(total_recon)
UCL_SPE = compute_KDE(total_recon,alpha=95) #kernel density estimation을 사용하여 상위 5%값 계산
print(f"SPE UCL: {UCL_SPE}")


# 5.2 SPE에 대한 UCL 확인
plt.figure(figsize=(10, 6))
plt.plot(total_recon)
plt.axhline(y=UCL_SPE, color='r', linestyle='--', label='UCL',linewidth=2)
plt.xlabel('Sample',fontsize=20)
plt.ylabel('Squared Prediction Error',fontsize=15)
plt.title(f'SPE for Train',fontsize=20)
plt.legend(fontsize=20)
plt.savefig(f'SPE for Train.png')
plt.show()



# 6. Hotelling's T2를 사용한 모니터링
# 6.1 T2의 UCL 계산
total_latent = VAE_encoder.predict(x_total)    #latent vector계산
mean = tf.reduce_mean(total_latent,axis=0)     #latent vector의 평균 계산
S, S_inv = compute_covariance_and_inverse(total_latent) #latent vector의 공분산과 공분산의 역행렬 계산
T2_total = compute_T2((total_latent-mean),S_inv)      #T2 Score 계산
T2_total = pd.DataFrame(T2_total)

UCL_T2 = compute_KDE(T2_total,alpha=95)   #kernel density estimation을 사용하여 상위 5%값 계산

print(f"T2 UCL: {UCL_T2}")

# 6.2 T2에 대한 UCL 확인
plt.figure(figsize=(10, 6))
plt.plot(T2_total)
plt.axhline(y=UCL_T2, color='r', linestyle='--', label='UCL',linewidth=2)
plt.xlabel('Sample',fontsize=20)
plt.ylabel('T2 score',fontsize=15)
plt.title(f'T2 score for Train',fontsize=20)
plt.legend(fontsize=20)
plt.savefig(f'T2 score for Train.png')
plt.show()



"""
fault 데이터를 사용한 모니터링 성능 평가 단계
"""
# 7. 21종류 Fault에 대해서 VAE모델의 모니터링 성능 평가
#모니터링 지표는 T2와 SPE를 사용하며, 모니터링 성능을 정량화하기 위해 F1 Score를 사용한다.

result = pd.DataFrame(columns=['Fault','FDR_SPE', 'FAR_SPE', 'F1_Score_SPE','FDR_T2', 'FAR_T2', 'F1_Score_T2'])

for i in range(1,22):
    fault = pd.read_csv(f'fault_{i}_test.csv',index_col=0)
    fault = scaler.transform(fault)
    fault = fault.astype('float32')

    # Fault 데이터의 SPE 계산
    fault_SPE = tf.reduce_sum(tf.square(fault-VAE.predict(fault)),axis=1)
    fault_SPE = pd.DataFrame(fault_SPE)

    no_fault_SPE = fault_SPE[:160]      #fault는 160번째 샘플부터 발생하기 때문에 0~160번째 샘플은 정상데이터
    yes_fault_SPE = fault_SPE[160:]     #fault는 160번째 샘플부터 발생하기 때문에 160번째 샘플은 이후부터 fault

    # Fault 데이터의 T2 계산
    fault_latent = VAE_encoder.predict(fault)  # latent vector계산
    fault_T2 = compute_T2((fault_latent - mean), S_inv)  # T2 Score 계산, 여기서 사용되는 mean과 S_inv는 train데이터로 계산된 값을 사용해야 함
    fault_T2 = pd.DataFrame(fault_T2)

    no_fault_T2 = fault_T2[:160]      #fault는 160번째 샘플부터 발생하기 때문에 0~160번째 샘플은 정상데이터
    yes_fault_T2 = fault_T2[160:]     #fault는 160번째 샘플부터 발생하기 때문에 160번째 샘플은 이후부터 fault

    # 모니터링 성능 정량화
    TP_SPE, FP_SPE, F1_SPE = compute_F1(fault_data=np.array(yes_fault_SPE), no_fault_data=np.array(no_fault_SPE), UCL=UCL_SPE)
    TP_T2, FP_T2, F1_T2 = compute_F1(fault_data=np.array(yes_fault_T2), no_fault_data=np.array(no_fault_T2), UCL=UCL_T2)
    result.loc[len(result)] = [i,TP_SPE, FP_SPE, F1_SPE,TP_T2, FP_T2, F1_T2]


    # 모니터링 그래프 그리기
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))  # 2개의 서브플롯 생성

    # SPE 그래프
    axs[0].plot(fault_SPE, label='SPE')
    axs[0].axhline(y=UCL_SPE, color='r', linestyle='--', label='UCL', linewidth=2)
    axs[0].set_xlabel('Sample', fontsize=15)
    axs[0].set_ylabel('SPE', fontsize=15)
    axs[0].set_title(f'Fault {i} SPE', fontsize=20)
    axs[0].legend(fontsize=15)

    # T2 그래프
    axs[1].plot(fault_T2, label='T2')
    axs[1].axhline(y=UCL_T2, color='r', linestyle='--', label='UCL', linewidth=2)
    axs[1].set_xlabel('Sample', fontsize=15)
    axs[1].set_ylabel('T2 Score', fontsize=15)
    axs[1].set_title(f'Fault {i} T2', fontsize=20)
    axs[1].legend(fontsize=15)

    plt.tight_layout()

    plt.savefig(f'Fault{i}_SPE_T2.png')
    plt.close()

result.to_csv('result.csv',index=False)


