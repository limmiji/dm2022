# 1. 데이터를 획득하고 모델을 적용하기 위한 준비과정
ucla = read.csv('http://stats.idre.ucla.edu/stat/data/binary.csv')
str(ucla)

ucla$admit = factor(ucla$admit) # factor형 변환
str(ucla)

#------------------------------------------------------------------#

# 2. 학습데이터와 테스트 데이터 분리과정
# (학습데이터 6, 테스트데이터 4)
n=nrow(ucla)
i = 1:n
train_list = sample(i, n*0.6)
test_list = setdiff(i, train_list)
ucla_train = ucla[train_list, ]
ucla_test = ucla[test_list, ]

#------------------------------------------------------------------#

# 3. 학습데이터로 모델을 만드는 과정(모델링)
library(caret)

# 결정트리
r = train(admit~., data=ucla_train, method = 'rpart')

# 랜덤포레스트(트리개수 50개)
f = randomForest(admit~., data=ucla_train, method = 'rf', ntree=50)

# 랜덤포레스트(트리개수 1000개)
f1 = randomForest(admit~., data=ucla_train, method = 'rf', ntree=1000)

# K-NN
k = train(admit~., data=ucla_train, method = 'knn')

# SVM(radial basis)
s = train(admit~., data=ucla_train, method = 'svmRadial')

# SVM(polynimial)
s1 = train(admit~., data=ucla_train, method = 'svmPoly')

#------------------------------------------------------------------#

# 4. 테스트데이터로 예측하고, 예측결과를 혼동행렬로 출력하는 과정
# 결정트리
r_pred = predict(r, newdata=ucla_test)
rc = confusionMatrix(r_pred, ucla_test$admit)
rc

# 랜덤포레스트 (트리개수 50개)
f_pred = predict(f, newdata=ucla_test)
fc = confusionMatrix(f_pred, ucla_test$admit)
fc

# 랜덤포레스트 (트리개수 1000개)
f1_pred = predict(f1, newdata=ucla_test)
f1c = confusionMatrix(f1_pred, ucla_test$admit)
f1c

# K-NN
k_pred = predict(k, newdata=ucla_test)
kc = confusionMatrix(k_pred, ucla_test$admit)
kc

# SVM(radial basis)
s_pred = predict(s, newdata=ucla_test)
sc = confusionMatrix(s_pred, ucla_test$admit)
sc

# SVM(polynomial)
s1_pred = predict(s1, newdata=ucla_test)
s1c = confusionMatrix(s1_pred, ucla_test$admit)
s1c

#------------------------------------------------------------------#

# 5. 혼동행렬로부터 정확도를 계산하는 과정
table(ucla_test$admit)

check_accuracy = function(x,y) {
  result = (x+y)/160
  cat("정확도 : ", result*100, "%")
}

check_accuracy(rc$table[1], rc$table[1,2])
check_accuracy(fc$table[1], fc$table[1,2])
check_accuracy(f1c$table[1],f1c$table[1,2])
check_accuracy(kc$table[1], kc$table[1,2])
check_accuracy(sc$table[1], sc$table[1,2])
check_accuracy(s1c$table[1], s1c$table[1,2])

