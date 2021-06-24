## R

#### 변수에 데이터 할당

```R
val <- c(1, 2, 3, 4, 5)
val <- c(1:5)
# 1 2 3 4 5


val <- seq(0, 100, 20)
# 0  20  40  60  80 100

# 벡터의 길이를 지정해서 생성 가능
m <- seq(0, 1000, length=5)
# 0  250  500  750 1000

# 벡터 항목 각각을 해당 횟수만큼 반복
part<-rep(seq(0, 100, 20), each=2)
#  0   0  20  20  40  40  60  60  80  80 100 100

# 벡터 전체를 해당 횟수만큼 반복
part<-rep(seq(0, 100, 20), times=2)
#  0  20  40  60  80 100   0  20  40  60  80 100

# 빈 벡터를 생성해 놓을 수 있음
a <- vector(mode="numeric", length=20)
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
a <- vector(mode="character", length=10)
# "" "" "" "" "" "" "" "" "" ""
```

#### 범주형 변수 생성

factor 이용

```R
width<-factor(c("18평", "24평", "24평", "33평", "33평", "33평", "18평"))
# 18평 24평 24평 33평 33평 33평 18평
# Levels: 18평 24평 33평

width<-factor(c("18평", "24평", "24평", "33평", "33평", "33평", "18평"), ordered=TRUE)
# 18평 24평 24평 33평 33평 33평 18평
# Levels: 18평 < 24평 < 33평
# 범주형 변수들에 순서를 부여

width <- factor(c("18평", "24평", "24평", "33평", "33평", "33평", "18평"), c("18평", "24평"))
# levels에 일부 값만 부여하면 그에 해당하는 값만 출력되고 아닌 값은 NA로 출력
# 18평 24평 24평 <NA> <NA> <NA> 18평
# Levels: 18평 24평

# 각 요소 count
# python value_count와 동일
table(width)
# 18평 24평 33평 
#  2    2    3 

unclass(width)
# 1 2 2 3 3 3 1
# attr(,"levels")
# "18평" "24평" "33평"
# 각 요소를 상수로 나타내고 각 상수에 매칭되는 값을 나열
```

#### 데이터 자료형 변경

- as.integer(), as.numeric(), as.character(), as.factor() 등으로 자료형 변경 가능



#### 리스트

- 여러가지 자료형을 모두 삽입이 가능

```R
myfavorit <- list(friend="홍길동", mynum=7, myalpha="z")
# $friend
# [1] "홍길동"
# $mynum
# [1] 7
# $myalpha
# [1] "z"

# $ 이후에 key 값을 명시해서 해당 value만 확인 가능
myfavorit$myalpha
# "z"

# 기존에 없던 항목도 추가 가능 (벡터도 추가 가능)
myfavorit$mysong <- "hello"
# $mysong
# [1] "hello"
```



#### 매트릭스

- 다차원 배열을 생성

```R
age <- matrix(c(25, 33, 32, 37, 27, 38), nrow=2)
#      [,1] [,2] [,3]
# [1,]   25   32   27
# [2,]   33   37   38

# byrow=TRUE : 행 중심으로 matrix를 구성
age <- matrix(c(25, 33, 32, 37, 27, 38), nrow=2, byrow=TRUE)
#      [,1] [,2] [,3]
# [1,]   25   33   32
# [2,]   37   27   38


# 행 중심으로 구성
info <- matrix(c("177cm", "68kg", "165cm", "57kg", "160cm", "55kg", "155cm", "50kg"), ncol=2, byrow=TRUE)
#      [,1]    [,2]  
# [1,] "177cm" "68kg"
# [2,] "165cm" "57kg"
# [3,] "160cm" "55kg"
# [4,] "155cm" "50kg"

# 각 행과 열에 이름을 지정 (앞에 나오는 부분이 행 이름, 뒤가 열 이름)
dimnames(info) <- list(c("1번", "2번", "3번", "4번"), c("키", "몸무게"))
#     키      몸무게
# 1번 "177cm" "68kg"
# 2번 "165cm" "57kg"
# 3번 "160cm" "55kg"
# 4번 "155cm" "50kg"

# 열 데이터 추가
info <- cbind(info, c("남", "남", "여", "여"))
#      키      몸무게     
# 1번 "177cm" "68kg" "남"
# 2번 "165cm" "57kg" "남"
# 3번 "160cm" "55kg" "여"
# 4번 "155cm" "50kg" "여"

a1 <- c("180cm", "70kg", "남")
a2 <- c("185cm", "68kg", "남")
# 행 데이터 추가
info <- rbind(info, a1, a2)
#      키      몸무게     
# 1번 "177cm" "68kg" "남"
# 2번 "165cm" "57kg" "남"
# 3번 "160cm" "55kg" "여"
# 4번 "155cm" "50kg" "여"
# a1  "180cm" "70kg" "남"
# a2  "185cm" "68kg" "남"

dimnames(info) <- list(c("1번", "2번", "3번", "4번", "5번", "6번"), c("키", "몸무게", "성별"))
#      키      몸무게 성별
# 1번 "177cm" "68kg" "남"
# 2번 "165cm" "57kg" "남"
# 3번 "160cm" "55kg" "여"
# 4번 "155cm" "50kg" "여"
# 5번 "180cm" "70kg" "남"
 #6번 "185cm" "68kg" "남"
```



#### 데이터프레임

- 여러 타입의 데이터 존재 가능
- as.data.frame() 메서드로 다른 형태의 데이터를 데이터프레임으로 변환
- data.frame() 메서드로 데이터프레임 생성

```R
# 기존의 파일을 읽어올 때
# 한글이므로 fileEncoding 지정
read.table("personal.txt", header=TRUE, fileEncoding="UTF-8")

pinfo
#     이름 거주지역 나이 성별
# 1 홍민수     인천   25   남
# 2 조정란     경기   33   여
# 3 국정수     서울   43   남
# 4 라윤정     서울   35   여
# 5 한주연     인천   37   남

lifeinfo
#           취미     관심사
# 1       목공예       취업
# 2   패러글라이딩 아파트분양
# 3         등산   주택대출
# 4       꽃꽃이       이직
# 5          독서 교통인프라

cbind(pinfo, lifeinfo)
#     이름 거주지역 나이 성별         취미     관심사
# 1 홍민수     인천   25   남       목공예       취업
# 2 조정란     경기   33   여 패러글라이딩 아파트분양
# 3 국정수     서울   43   남         등산   주택대출
# 4 라윤정     서울   35   여       꽃꽃이       이직
# 5 한주연     인천   37   남         독서 교통인프라

otherinfo 
#      이름 거주지역 나이 성별
# 1 황성주     강원   42   여
# 2 윤준영     충북   45   여

rbind(pinfo, otherinfo)
#     이름 거주지역 나이 성별
# 1 홍민수     인천   25   남
# 2 조정란     경기   33   여
# 3 국정수     서울   43   남
# 4 라윤정     서울   35   여
# 5 한주연     인천   37   남
# 6 황성주     강원   42   여
# 7 윤준영     충북   45   여
```

- cbind : 열 기준 병합 (열 추가)
  - axis=0
- rbind : 행 기준 병합 (행 추가)
  - axis=1

```R
pinfo
#     이름 거주지역 나이 성별
# 1 홍민수     인천   25   남
# 2 조정란     경기   33   여
# 3 국정수     서울   43   남
# 4 라윤정     서울   35   여
# 5 한주연     인천   37   남
pjobinfo
#     이름     직업
# 1 홍민수     학생
# 2 조정란   공무원
# 3 박장곤   회사원
# 4 한지수   자영업
# 5 한주연 프리랜서
# 6 주민정     학생

merge(pinfo, pjobinfo)
#      이름 거주지역 나이 성별     직업
# 1 조정란     경기   33   여   공무원
# 2 한주연     인천   37   남 프리랜서
# 3 홍민수     인천   25   남     학생

merge(pinfo, pjobinfo, all=TRUE)
#     이름 거주지역 나이 성별     직업
# 1 국정수     서울   43   남     <NA>
# 2 라윤정     서울   35   여     <NA>
# 3 박장곤     <NA>   NA <NA>   회사원
# 4 조정란     경기   33   여   공무원
# 5 주민정     <NA>   NA <NA>     학생
# 6 한주연     인천   37   남 프리랜서
# 7 한지수     <NA>   NA <NA>   자영업
# 8 홍민수     인천   25   남     학생
```

- merge : 공통되는 열을 키값으로 해당되는 데이터들만 출력 (inner join)
- all=TRUE : 공통되는 키값이 없어도 모든 데이터 출력 (outer join)

```R
subset(pinfo, select=-거주지역)
#      이름 나이 성별
# 1 홍민수   25   남
# 2 조정란   33   여
# 3 국정수   43   남
# 4 라윤정   35   여
# 5 한주연   37   남

subset(pinfo, select=c(이름, 나이))
#     이름 나이
# 1 홍민수   25
# 2 조정란   33
# 3 국정수   43
# 4 라윤정   35
# 5 한주연   37

# 다음과 같이 필요한 부분만 추출 가능
pinfo[c(2:4), -c(3:5)]
#     이름 거주지역
# 2 조정란     경기
# 3 국정수     서울
# 4 라윤정     서울

# 다음과 같이 column명과 index(row)명 지정
colnames(data1) <- c("성명", "나이", "성별")
rownames(data1) <- c("가", "나", "다", "라", "마")
```

- subset : 데이터프레임에서 필요한 부분만 추출
  - -가 앞에 붙으면 그 열을 제외

#### 조건문

```R
a <- c(10, 13, 7, 8, 50)
ifelse(a%%2==0, "짝수", "홀수")
# "짝수" "홀수" "홀수" "짝수" "짝수"
```

- ifelse : 여러 데이터를 한번에 조건 처리가능

#### 반복문

```R
evensum <- 0
for (x in 1:length(a))
{
  if(a[x]%%2 == 0) evensum <- evensum + a[x]
  cat("evensum :", evensum, "\n")
}
# evensum : 10 
# evensum : 10 
# evensum : 10 
# evensum : 18 
# evensum : 68 
evensum
# 68
```

- cat : print 역할

```python
a <- c(23, 15, 17, 33, 45)
i <- 1
repeat{
  if(a[i] < mean(a)) break
  i < i + 1
}
a[i]
# 23
mean(a)
# 26.6
```

- repeat : while (TRUE)와 동일
  -  조건문에 따른 break가 필수



#### 함수

**apply**

```R
#   신장 체중
# 1  177   NA
# 2  180 77.3
# 3  167 80.0
# 4  165 60.0
# 5  170   NA
# 6   NA 64.0

# 열 중심으로 결측치를 제외한 평균치를 구함 (행 중심은 1)
apply(df, 2, mean, na.rm=TRUE)
#     신장    체중 
# 171.800  70.325 

# 계산결과를 list형태로 변환
lapply(df, max, na.rm=TRUE)
# $신장
# [1] 180
# $체중
# [1] 80

# 계산 결과를 벡터형태로 변환
sapply(df, max, na.rm=TRUE)
# 신장 체중 
# 180   80 

#     나이 성별 평점
# 1   33   남  4.3
# 2   28   남  4.2
# 3   35   여  4.1
# 4   29   여  3.7
# 5   36   남  4.5
# 6   32   여  4.4
# 7   30   여  3.8

# 그룹별로 함수를 적용가능 (성별에 따른 평점 평균)
tapply(dflist$평점, dflist$성별, mean)
#        남       여 
# 4.333333 4.000000 

# 성별에 따른 나이 평균
# simplify=FALSE : 결과를 리스트 형태로 반환
tapply(dflist$나이, dflist$성별, mean, simplify=FALSE)
# $남
# [1] 32.33333
# $여
# [1] 31.5
```

**사용자정의 함수**

```python
basicst <- function(x)
{
  amin <- min(x)
  amax <- max(x)
  amean <- mean(x)
  avar <- var(x)
  astd <- sd(x)
  totinfo <- list(최소값=amin, 최대값=amax, 평균=amean, 분산=avar, 표준편차=astd)
  return(totinfo)
}

a
# 23 15 17 33 45

basicst(a)
# $최소값
# [1] 15
# $최대값
# [1] 45
# $평균
# [1] 26.6
# $분산
# [1] 154.8
# $표준편차
# [1] 12.44186

source("파일 경로")
```

- source : 외부 파일에 정의해놓은 함수 로드



#### 파일 불러오기

```R
scan("파일명", what="자료형", sep="seperator")

# 다음과 같이 사용하면 공백이 입력되기 전까지 변수를 입력받음
x <- scan()

# 줄 단위로 데이터 로드
readLines("파일명")

# 엑셀 파일을 읽어오는 패키지
install.packages("xlsx")
require(xlsx)
df <- read.xlsx2("subway.xlsx", sheet=1)
df
#   구.분       역명    X1월    X2월    X3월     총계 일평균
# 1 1호선  서울역(1) 4126245 3661950 4145729 11933924 132599
# 2 1호선    시청(1) 1499505 1229076 1493112  4221693  46908
# 3 1호선       종각 3039562 2477861 2924326  8441749  93797
# 4 1호선 종로3가(1) 2435003 1999718 2290837  6725558  74728
# 5 1호선    종로5가 1758749 1488469 1791087  5038305  55981


# ctrl+c로 일부 부분만 clipboard에 저장해놓고 불러올 수 있음
a <- readClipboard()
a
df <- read.table(file="clipboard", sep="\t", header=TRUE)
df
#    구.분       역명        X1월        X2월        X3월         총계    일평균
# 1  1호선  서울역(1)  4,126,245   3,661,950   4,145,729   11,933,924   132,599 
# 2  1호선    시청(1)  1,499,505   1,229,076   1,493,112    4,221,693    46,908 
# 3  1호선       종각  3,039,562   2,477,861   2,924,326    8,441,749    93,797 
# 4  1호선 종로3가(1)  2,435,003   1,999,718   2,290,837    6,725,558    74,728 
# 5  1호선    종로5가  1,758,749   1,488,469   1,791,087    5,038,305    55,981 
# 6  1호선  동대문(1)  1,032,643     942,226   1,141,235    3,116,104    34,623 
# 7  1호선  신설동(1)  1,016,609     861,509   1,114,369    2,992,487    33,250 
# 8  1호선     제기동  1,273,577   1,164,476   1,319,059    3,757,112    41,746 
# 9  1호선     청량리  1,898,670   1,683,252   1,973,381    5,555,303    61,726 
# 10 1호선     동묘앞    591,418     549,158     686,803    1,827,379    20,304 
```

- readClipboard : 클립보드에 있는 데이터 확인



#### 그래프

- main : 그래프 제목
- sub : 그래프 부 제목
- xlab, ylab : x, y축 제목
- type : plot 형태
- axes : plot의 테두리선
- col : plot의 색상

```python
plot.new()
x <- c(1:5)
y <- c(1:5)
# mfrow : 행과 열 값을 순서대로
par(mfrow=c(2, 3))
kind <- c("p", "b", "l", "o", "s", "h")
for(i in 1:length(kind))
{
  # paste : type=과 뒤의 값을 붙여줌
  title <- paste("type=", kind[i])
  plot(x, y, main=title, type=kind[i], col=rainbow(length(kind)))
}
```

<img src="https://user-images.githubusercontent.com/58063806/123093452-ac747c80-d466-11eb-8c70-2c7d4a692e76.png" width=100% />

```python
plot(1:5, 1:5, main="여러 형태의 점")
points(3.2, pch=2, cex=2)
points(2, pch=15, cex=3)
points(2.8, pch=20, cex=1)
```

- pch : 점의 형태

<img src="https://user-images.githubusercontent.com/58063806/123094505-ebef9880-d467-11eb-9e37-7a14fd0dafdf.png" width=25% />

- cex : 점의 크기

<img src="https://user-images.githubusercontent.com/58063806/123094294-b054ce80-d467-11eb-9ac8-b177c88ee337.png" width=50% />

```python
plot(1:5, 1:5, type="n")
lines(c(1,3), c(3,3), lty="dotted", lwd=3, col="red")
lines(c(1,3), c(4,4), lty="solid", lwd=2, col="green")
lines(c(1,5), lty="dashed", lwd=4, col="blue")
```

- lty : line 타입
- lwd : 선의 굵기

<img src="https://user-images.githubusercontent.com/58063806/123096226-d67b6e00-d469-11eb-8596-54c35fa0865c.png" width=50% />

```python
# 그래프를 새창에 출력
dev.new(width=10, height=10, unit="in")
plot.new()
height <- c(165, 170, 173, 175, 180, 176, 172, 168)
weight <- c(66, 70, 72, 80, 85, 78, 65, 62)
irum <- c("a", "b", "c", "d", "e", "f", "g", "h")
plot(height, weight, type="b", pch=21, col=rainbow(length(height)))
abline(h=seq(70, 80, 5), col="gray", lty=2)
text(175, 73, "표준", col="green")
axis(2, at=height, labels=height, col.axis="gray")
legend("bottomright", col=rainbow(length(height)), legend=irum, lty=1)
```

- abline : figure에 선 추가
- text : 해당 위치에 글자 삽입
- axis : 그래프에 축 삽입
- legned : 범례 삽입

<img src="https://user-images.githubusercontent.com/58063806/123102721-50165a80-d470-11eb-8c76-984d2efd7220.png" width=60% />

```R
dev.new(width=10, height=10, unit="in")
plot.new()
y1 <- c(0.8, 0.5, 0.4, 0.4, 0.5, 0.7)
y2 <- c(0.8, 1.3, 1.0, 1.3, 0.9, 1.2)
x <- c(1:6)
# 그래프 중첩하기 위함
par(new=TRUE)
plot(x, y1, ylab="소비자물가상승률", type="o", col='red', ylim=c(0.3, 1.5))
# 그래프 중첩하기 위함
par(new=TRUE)
plot(x, y2, lty="dotted", ylab="소비자물가상승률", type="l", 
     col="blue", ylim=c(0.3, 1.5))
legend(locator(1), legend=c("2015년", "2016년"), lty=1, 
       bg="yellow", col=c("red", "blue"))
```

- legend의 locator : 클릭하는 지점에 범례 생성

<img src="https://user-images.githubusercontent.com/58063806/123104598-0d558200-d472-11eb-8166-6acc8c558f3a.png" width=50% />

**bar**

```python
# 각 행별로 남자, 여자 열의 값을 합함
rowSums(data[, c("남자", "여자")], na.rm=TRUE)
# 기존 data에 열 기준 병합
data <- cbind(data, tot)

dev.new(width=10, height=10, unit="in")
plot.new()
data1 <- subset(data, tot >= 500000)
data1
#     자치구   남자   여자    tot
# 11 노원구 281538 296683 578221
# 12 은평구 244964 257614 502578
# 16 강서구 291216 304475 595691
# 21 관악구 266773 262258 529031
# 23 강남구 279209 302551 581760
# 24 송파구 325950 341530 667480

# 1 ~ 6행, 2 ~ 3열 부분을 matrix로 변환
barplot(as.matrix(data1[1:6, 2:3]), legend=c("남", "여"), 
        las=1, col=c('darkgreen', "pink"), beside=TRUE, main="인구 50만 이상 성별현황", ylim=c(0, 400000))
```

- besied=TRUE : 그래프가 겹쳐지지 않도록 하기 위함

<img src="https://user-images.githubusercontent.com/58063806/123108440-64a92180-d475-11eb-8fb8-9f66892fb8f0.png" width=50% />

**histogram**

```R
x <- c(23, 33, 32, 45, 37, 28, 15, 35, 43, 27, 46, 33, 38, 46, 50 ,25)
hist(x, main="연령분포", xlim=c(15, 50), col="yellow")
```

<img src="https://user-images.githubusercontent.com/58063806/123109486-3f68e300-d476-11eb-8fbf-a093dd5b2a80.png" width=50% />

**boxplot**

```R
#        월별 출생 사망
# 1  2013-01 8530 3830
# 2  2013-02 7045 3464
# 3  2013-03 7316 3809
# 4  2013-04 7162 3417
# 5  2013-05 6843 3469
# ...

boxplot(data$출생, data$사망, names=c("출생", "사망"), 
        col=c("pink", "darkgreen"), main="서울 2013-2014 출생사망자 비교")
```

<img src="https://user-images.githubusercontent.com/58063806/123111254-bc488c80-d477-11eb-90f5-ff9a1da269d8.png" width=50% />



#### ggplot

```R
#   mat eng avg irum
# 1  55  65  53   김
# 2  75 100  70   이
# 3  80  45  83   박
# 4  65  50  70   최
# 5  90  75  93   문
# 6 100  90  95   윤
# 7  70  90  75   손
# 8  85  65  80   정

dev.new(width=10, height=10, unit="in")
ggplot(df, aes(mat, avg))+xlab("score")+ylab("avg_score")+
  geom_line(color="red")+geom_point(color="yellow")+
  geom_line(aes(eng, avg), color="darkgreen")+
  geom_point(aes(eng, avg), color="blue")
```

- ggplot은 레이어를 추가하면 이후 레이어에 상속
  - 원본 ggplot이 아닌 이후의 layer에서 만들어진 내용은 상속이 안됨

<img src="https://user-images.githubusercontent.com/58063806/123208927-dd05f600-d4fa-11eb-86c3-951ec93ca6ed.png" width=60% />

```R
#     차종   선별   출발지   도착지   거리 		총운행횟수 총이용인원 이용율
# 1   우등   88선     광주     울산 327.8        412       7283   63.1
# 2   고속   88선     광주     울산 327.8        145       3050   46.7
# 3   우등   88선     광주 울산신복 327.8        164        545   11.9
# 4   고속   88선     광주 울산신복 327.8         70        311    9.9
# 5   우등   88선     광주   동대구 219.3       1369      21873   57.1
# ...
dev.new(width=10, height=10, unit="in")
ggplot(datainfo, aes(총운행횟수, 이용율))+geom_point(aes(color=선별, size=거리))
```

- 그룹별로 시각화
- 자동적으로 범례 생성됨

<img src="https://user-images.githubusercontent.com/58063806/123209401-7f25de00-d4fb-11eb-8b38-ae4596876bdd.png" width=60%/>

```R
dev.new(width=10, height=10, unit="in")
ggplot(datainfo, aes(선별, 총이용인원))+geom_bar(stat="identity", fill="orange")
```

- geom_bar의 stat요소는 기본적으로 stat_bin (개수 카운트)으로 설정
- 항목의 계산을 위해서는 stat="identity"로 설정
- fill : 막대 그래프 내부 색, color : 막대 그래프 테두리 색 

<img src="https://user-images.githubusercontent.com/58063806/123209585-cd3ae180-d4fb-11eb-8533-79d42de1044d.png" width=60%/>

```R
meandf <- as.data.frame(with(datainfo, tapply(이용율, 선별, mean, na.rm=TRUE)))
meandf$노선 <- rownames(meandf)
names(meandf) <- c("이용율", "노선")
meandf
#          이용율   노선
# 88선   44.66667   88선 
# 경부선 48.20476 경부선
# 경인선 54.91600 경인선
# 구마선 46.95000 구마선
# ...
dev.new(width=10, height=10, unit="in")
ggplot(meandf, aes(노선, 이용율))+geom_bar(stat="identity", aes(fill=노선))
```

- 노선에 따른 평균이용율

<img src="https://user-images.githubusercontent.com/58063806/123212547-ea71af00-d4ff-11eb-9b3b-d37d0011451c.png" width=50% />

```R
meandf1 <- as.data.frame(with(datainfo, aggregate(이용율, list(선별, 차종), 
                                                    mean, na.rm=TRUE)))
colnames(meandf1) <- c("선별", "차종", "평균이용율")
meandf1
#      선별 차종 평균이용율
# 1    88선 고속   37.70000
# 2  경부선 고속   48.91930
# 3  경인선 고속   61.19091
# ...
# 9    88선 우등   49.10000
# 10 경부선 우등   47.35625
# 11 경인선 우등   49.98571
# ...
dev.new(width=10, height=10, unit="in")
ggplot(meandf1, aes(선별, 평균이용율))+geom_bar(stat="identity", aes(fill=차종))
```

- aggregate : 그룹핑 기준이 되는 컬럼이 여러개 일때 list의 형태로 전달해서 사용
- 노선과 차종에 따른 평균이용율

<img src="https://user-images.githubusercontent.com/58063806/123213210-b2b73700-d500-11eb-87dc-0897b3763231.png" width=60%/>

```R
ggplot(meandf1, aes(선별, 평균이용율))+
geom_bar(stat="identity", aes(fill=차종), position="fill")
```

- 100% 기준으로 각 노선별 차종에 따른 평균이용율 비교
  - position="fill"

<img src="https://user-images.githubusercontent.com/58063806/123213310-d1b5c900-d500-11eb-9b91-5fe6a5706431.png" width=60% />

```R
ggplot(meandf1, aes(선별, 평균이용율))+
geom_bar(stat="identity", aes(fill=차종), position="dodge")+
geom_text(aes(y=평균이용율, label=평균이용율), color="black", size=2)
```

- 각 노선별 차종에 따른 평균이용율 비교 (차종에 따라 나누어서 시각화)
  - position="dodge"

<img src="https://user-images.githubusercontent.com/58063806/123213503-09bd0c00-d501-11eb-9555-c05876338ca5.png" width=60% />