reshape, squeeze, expand_dims 
1) reshape은 원하는 shape를 직접 입력하여 바꿀 수 있다.

   특히 shape에 -1를 입력하면 고정된 차원은 우선 채우고 남은 부분을 알아서 채워준다.



2) squeeze는 차원 중 사이즈가 1인 것을 찾아 스칼라값으로 바꿔 해당 차원을 제거한다.



3) expand_dims는 axis로 지정된 차원을 추가한다.

https://m.blog.naver.com/PostView.nhn?blogId=wideeyed&logNo=221164750517&proxyReferer=https%3A%2F%2Fwww.google.co.kr%2F