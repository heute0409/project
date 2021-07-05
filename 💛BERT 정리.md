# 💛BERT

[https://www.youtube.com/watch?v=qlxrXX5uBoU](https://www.youtube.com/watch?v=qlxrXX5uBoU)

### 자연어 처리 NLP

- 자연어 처리의 종류, 현업에 적용된 예제
- BERT 탄생 과정, 적용 메커니즘
- BERT를 이용한 자연어 처리 실험 결과
- 오픈 소스로 공개된 BERT 코드를 이용해 한국어 BERT를 학습, 다양한 자연어 처리 task 실습

## 자연어

Symbolic approach : 규칙/지식 기반 접근법 

패턴 별로 인식하게 함. 

Statical approach : 확률/통계 기반 접근법

TF-IDF를 이용한 키워드 추출 

단순히 빈도만으로 추출하는 게 아니라(이유 : 그리고, 그러나 등 접속사가 많을 수도 있으므로) 주요한 키워드를 인식해 추출

### 자연어 처리 과정

**전처리** 

학습 데이터의 양이 단순히 많다고 좋은 게 아니라, 양질의 데이터를 쌓아야 한다. 

**Tokenizing**

자연어를 어떤 단위로 자른다. 

**Applications**

![%F0%9F%92%9BBERT%209a1c7c87bcb84c65805a552852d86964/Untitled.png](%F0%9F%92%9BBERT%209a1c7c87bcb84c65805a552852d86964/Untitled.png)

대부분의 자연어 처리 문제는 분류 문제이다. 

**embedding** 

자연어를 좌표평면 위에 나열하는 것

**WordtoVec** 

단어가 가진 의미를 다차원에 배열. 중심 단어 주변 단어를 이용해 중심 단어를 추론하는 방식으로 학습

![%F0%9F%92%9BBERT%209a1c7c87bcb84c65805a552852d86964/Untitled%201.png](%F0%9F%92%9BBERT%209a1c7c87bcb84c65805a552852d86964/Untitled%201.png)

장점 : 단어간의 유사도 측정, 관계 파악에 용이. 벡터 연산을 통한 추론이 가능. 

단점 : subword(ex. 서울시의 '시') information을 무시한다. 

**fasttext**

facebook에서 공개한 오픈소스 라이브러리 

n-gram으로 단어를 분리한 후, 모든 n-gram vector를 합산한 후 평균을 통해 단어 벡터 획득 

오탈자에 대해서도 본래 단어와 유사한 n-gram이 많아, 유사한 단어 벡터 획득 가능. OOV (out of vocabulary)에 대한 처리가 가능하다. 

**OOV**

NLP에서 빈번히 발생하는 데이터 문제로 input language가 database 혹은 input of embedding에 없어서 처리를 못 하는 것.

이 문제를 해결하기 위해 embedding 방식으로 word embedding이 아니라 character embedding이 사용되기도 한다.

Word2vec이나 fasttext와 같은 word embedding 방식은 동형어, 다의어 등에 대해선 성능이 좋지 못하다. 

### Word embedding 방식의 한계

- 문맥 확인이 불가능하다

**Markov 확률 기반의 언어 모델**

문장에서 단어에 대한 확률값을 계산한다. 이렇게 학습하다 보면 문장도 확률 계산이 가능해진다. 

### **RNN (Recurrent Neural Network) 기반의 언어 모델**

히든 노드가 방향을 가진 엣지로 연결돼 순환구조를 이루는 인공신경망의 한 종류

이전 state 정보가 다음 state를 예측하는데 사용됨으로써 시계열 데이터 처리에 특화 

앞의 문맥을 고려한 output을 얻을 수 있다. 

→ 문맥을 인코딩이 가능해진다. 

**Seq2Seq** 

챗봇으로도 이용한다. 

### RNN의 구조적 문제점

앞선 sequence를 전달하기 때문에 정보가 점차 희석이 된다. 

최종 output인 context vector는 고정적인 값이 나온다. 

쓸데 없는 token들까지 중요 단어로 선정될 수 있다. 

⇒ Attention 모델이 등장!

### Attention 모델

핵심 메커니즘 : 특정한 단어에 집중을 줘서 번역을 하자! 

**기존 RNN과 다른 점?**

각각 셀에서 나온 output vector를 사용하자. 

디코더에서 dynamic하게 context vector를 얻어내자 

히든 레이어의 score에 softmax를 취한다. 이것을 attention weight로 결정한다. 

attention weight와 hidden state를 곱해서 context vector를 구한다. 이 vector가 decoder로 들어가게 된다. 

decoder의 hidden state가 attention weight 계산에 영향을 준다. 

![%F0%9F%92%9BBERT%209a1c7c87bcb84c65805a552852d86964/Untitled%202.png](%F0%9F%92%9BBERT%209a1c7c87bcb84c65805a552852d86964/Untitled%202.png)

**attention for neural machine translation (NMT)**

**attention for speech to text (STT)** 

특정 프레임에 대해서 attention을 강하게 연결해서 학습이 가능해진다. 시각화가 가능해진다. 

- **문맥에 따라 동적으로 할당되는 encode의 attention weight로 인한 dynamic context vector를 획득.**
- **기존 Seq25eq의 encoder, decoder 성능을 비약적으로 향상시킴.**

단점 : 연산 속도가 느리다. 

→ 시계열 단계로 이루어지는 학습이기 때문에 앞 state의 연산이 끝나야 한다. 

⇒ self-attention의 등장 

### self-attention

decoder 결과가 정답과 다르면 context vector를 고치기 위해서 FC와 attention weight를 조절했었다. 

RNN + attention에 적용된 attetion은 decoder가 해석하기에 가장 적합한 weight를 찾고자 노력하는 것이다. 

→ attention이 decoder가 아니라, input인 값을 가장 잘 표현할 수 있도록 학습하면? 자기 자신을 가장 잘 표현할 수 있는 좋은 embedding이 가능하지 않을까? → 이러한 관점에서 self attention이 등장했다.

![%F0%9F%92%9BBERT%209a1c7c87bcb84c65805a552852d86964/Untitled%203.png](%F0%9F%92%9BBERT%209a1c7c87bcb84c65805a552852d86964/Untitled%203.png)

얻고자 하는 context vector = Query 

query*key = score → softmax → 정수값 

⇒ self-attention 의 attention weight

정수값*value = 각각의 중요도 값이 나온다. 

⇒ 각 단어가 갖는 의미를 context vector로 표현하게 된다. 

FC : fully connected feed forward network

![%F0%9F%92%9BBERT%209a1c7c87bcb84c65805a552852d86964/Untitled%204.png](%F0%9F%92%9BBERT%209a1c7c87bcb84c65805a552852d86964/Untitled%204.png)

Multi-head Self attention encoder 모델. Query, key, value로 구성된 attention layer를 동시에 여러 개 수행한다. 최종적으로 자기 자신을 표현하는 vector(context vector)를 획득하게 된다. 

BERT의 경우에는, 저 encoder를 많이 쌓는다. 

![%F0%9F%92%9BBERT%209a1c7c87bcb84c65805a552852d86964/Untitled%205.png](%F0%9F%92%9BBERT%209a1c7c87bcb84c65805a552852d86964/Untitled%205.png)

## BERT

RNN은 언어 위에 classification layer를 부착하여 task를 수행한다. BERT도 마찬가지로 이 모델 위에 classification layer를 부착하여 다양한 NLP task를 수행한다. 

WordPiece tokenizing을 사용한다. 

Q. BERT와 언어 모델은 동일한가?

→ 언어 자체를 수학적으로 모델링하기 위해 만든 것이고, BERT는 대표적인 언어 모델 중 하나이다. 

Q. Query key value는 다 다른 벡터인가?

→ 전부 다른 벡터이다. 

Q. 각 attention을 차별하기 위해 transformer에서 특별히 쓰이는 방법이 있는가?

→ 같은 작업을 여러 multi head attention으로 나눠함으로써 앙상블의 효과를 낸 것 같다. 

### WordPiece tokenizing

빈도수에 기반해 단어를 의미있는 패턴으로 잘라서 tokenizing한다. 

1. 학습 데이터를 이용해 vocab을 만들기 시작한다. 모든 캐릭터 단위로 tokenizing을 진행한다. 
2. 빈도수가 많이 등장하는 캐릭터 묶음을 계속 확인한다. 특정 vocab을 저장한다. 

![%F0%9F%92%9BBERT%209a1c7c87bcb84c65805a552852d86964/Untitled%206.png](%F0%9F%92%9BBERT%209a1c7c87bcb84c65805a552852d86964/Untitled%206.png)

![%F0%9F%92%9BBERT%209a1c7c87bcb84c65805a552852d86964/Untitled%207.png](%F0%9F%92%9BBERT%209a1c7c87bcb84c65805a552852d86964/Untitled%207.png)

##을 붙이는 이유는 같은 글자라고 같은 의미를 가지는 게 아니라, 위치에 따라 차이가 있다고 가정하는 것이다. 

![%F0%9F%92%9BBERT%209a1c7c87bcb84c65805a552852d86964/Untitled%208.png](%F0%9F%92%9BBERT%209a1c7c87bcb84c65805a552852d86964/Untitled%208.png)

vocab 후보를 정한다. 

![%F0%9F%92%9BBERT%209a1c7c87bcb84c65805a552852d86964/Untitled%209.png](%F0%9F%92%9BBERT%209a1c7c87bcb84c65805a552852d86964/Untitled%209.png)

창살 → 은 빈도수가 가장 높다. 이것은 best bi-gram pair이다. 

**Masked language model(MLM)** 

input token을 일정 확률로 masking 

ex. 15%로 토큰 하나를 선택 → 이 토큰의 80%를 마스킹, 10%를 랜덤하게 다른 토큰으로 바꾸고, 나머지 10%로 아무것도 하지 않는다. 

### NLP 실험 : GLUE datasets

![%F0%9F%92%9BBERT%209a1c7c87bcb84c65805a552852d86964/Untitled%2010.png](%F0%9F%92%9BBERT%209a1c7c87bcb84c65805a552852d86964/Untitled%2010.png)

### Sentence pair classification

두 가지 문장을 넣고, 어떤 관계를 가지느냐를 표현한다. A문장 다음에 B문장이 자연스럽게 이어지거나, 의미가 유사한지를 분류하기 위해 사용한다. 

### Single sentence pair classification

single 문장을 input으로 넣고 분류한다. ex. 영화 리뷰 문장을 넣고 이게 긍정인지 부정인지 분류한다. 

### Question and answering (SQuAD)

질문과 질문을 포함한 답변의 paragraph를 넣고 답변을 찾아낸다.

답변이 text로 나오는 게 아니라 paragraph의 index값을 출력하게 된다. (ex. 1번째~16번째 토큰이 정답이다.)

### Single sentence tagging

각각의 토큰에 대하여 출력을 나타낸다. 

문장의 토큰에 대한 정보값을 내는 것이다. 개체명, 기관명, 위치 등등. BIO tagging. 

⇒ 우리 프로젝트! 

멀티...로 했을 때 캐릭터 단위로 word를 tokenizing하면 성능이 오른다.