# 제품 관련 질의응답 챗봇
- 기능: 아마존 제품 관련 질의 응답 및 제품 정보 제공
- 구현: 학습된 질문-대답 데이터셋을 이용해 사용자가 입력한 질문에 대한 답변과 제품 정보(제품의 이름, 가격 정보 등)를 제공한다.
- 사용한 데이터셋: Amazon question/answer data, Amazon product data(Review data, Meta data)
                  http://jmcauley.ucsd.edu/data/amazon/
- 사용한 모델: Doc2Vec
- 문장 단위의 유사도 판별에서 높은 정확도를 갖는 Doc2Vec 모델을 이용해 사용자가 입력한 질문과 유사도가 높은 질문을 데이터셋에서 찾는다. 이후 해당 질문과 연결된 답변을 사용자에게 제공한다.
