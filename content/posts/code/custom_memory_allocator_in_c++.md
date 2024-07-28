+++
title = 'Custom Memory Allocator in C++'
date = 2024-07-28T10:11:26+09:00
draft = false
+++

C++에서 사용자가 직접 메모리 할당자를 만들려면 아래 코드를 사용하면 됩니다.

~~~c++
#include <cstdio>
#include <new>
 
// 사용자 정의 allocator 만들기
// => C++ 표준에서 정의한 규칙을 반드시 따라야 합니다.
template<typename T> 
class MyAlloc
{
public:
	// 규칙 1. 아래 3개의 멤버도 필요 합니다.(관례적인 코드)
	using value_type = T;
	MyAlloc() = default;
	template<typename U> MyAlloc(const MyAlloc<U>& ) {}

	// 규칙 2. 4개의 멤버 함수를 만들어야 합니다.
	T* allocate(std::size_t sz)
	{
		void* p = malloc(sizeof(T) * sz);
		printf("allocate %ld cnt, %p\n", sz, p);
		return static_cast<T*>(p);
	}
	void deallocate(T* p, std::size_t sz)
	{
		printf("deallocate %ld cnt, %p\n", sz, p);
		free(p);
	}
    /*  생성자와 소멸자는 경우에 따라 없어도 됩니다.
	template<typename ... ARGS>
	void construct(T* p, ARGS&& ... args)
	{
		printf("call constructor %p\n", p);
		new(p) T(std::forward<ARGS>(args)...);
	}
	void destroy(T* p)
	{
		printf("call destructor %p\n", p);
		p->~T();
	}
    */
};

// 규칙 3. == 와 != 연산자 제공되어야 합니다.
template<typename T> 
bool operator==(const MyAlloc<T>& a1, const MyAlloc<T>& a2)
{
	return true;
}
template<typename T>
bool operator!=(const MyAlloc<T>& a1, const MyAlloc<T>& a2)
{
	return false;
}
~~~

위 코드를 STL 컨테이너 queue에 사용해 보았습니다.
~~~c++
template<class T, class Container = std::deque<T>>
class queue;

template<class T, class Allocator = std::allocator<T>>
class deque;
~~~

queue는 기본 저장 컨테이너로 dequeu를 사용합니다.

~~~c++
#include <queue>

int main() {
    std::queue<int, std::deque<int, MyAlloc<int>>> q;
    q.push(1);
    q.push(2);
    q.push(3);

    return 0;
}
~~~

실행 결과는 아래와 같습니다.
~~~
allocate 1 cnt, 0x600003378000
allocate 1024 cnt, 0x138009200
call constructor 0x600003378000
call constructor 0x138009200
call constructor 0x138009204
call constructor 0x138009208
call destructor 0x138009200
call destructor 0x138009204
call destructor 0x138009208
deallocate 1024 cnt, 0x138009200
call destructor 0x600003378000
deallocate 1 cnt, 0x600003378000
~~~

맨 처음 1개의 공간을 동적할당하고 그 후에 더 입력되면 한꺼번에 1024개의 T형 메모리를 할당받습니다.
너무 많은 메모리가 불필요하게 할당되었습니다.
그리고 필요한 만큼 생성자를 호출합니다. 우리는 push를 3번 했으니 3번의 생성자가 호출되었습니다.

queue의 기본 저장 컨테이너로 list를 사용해 보았습니다.
~~~c++
template<class T, class Allocator = std::allocator<T>>
class list;
~~~
list의 타입은 위와 같습니다.
템플릿 두번째 인자로 위에서 구현한 메모리 할당자를 넣으면 됩니다.

~~~c++
#include <list>
#include <queue>

int main() {
    std::queue<int, std::list<int, MyAlloc<int>>> q;
    q.push(1);
    q.push(2);
    q.push(3);

    return 0;
}
~~~

실행 결과는 다음과 같습니다.
~~~
allocate 1 cnt, 0x600002e191c0
allocate 1 cnt, 0x600002e191a0
allocate 1 cnt, 0x600002e19220
deallocate 1 cnt, 0x600002e191c0
deallocate 1 cnt, 0x600002e191a0
deallocate 1 cnt, 0x600002e19220
~~~

list는 필요시마다 메모리를 할당요청합니다. 메모리가 낭비되진 않지만 한꺼번에 많은 메모리를 할당한 후 사용하는 것보다 속도측면에서 느릴 수 있습니다.
지난 번에 정해진 사이즈의 [FastMemoryPool]({{< ref "fast_memory_pool" >}}) 을 만들었습니다. 이것을 메모리 할당자에 사용하면 우리가 원하는 만큼의 메모리만 할당해서 사용할 수 있습니다.

MyAlloc 클래스를 아래와 같이 수정합니다. 아래 커스텀 메모리 할당자는 list만을 위한 것이므로 sz 파라미터를 이용하지 않습니다.
~~~c++
#include <cstdio>
#include <new>
 
template<typename T, int N> 
class MyAlloc
{
public:
	using value_type = T;
	MyAlloc() = default;
	template<typename U> MyAlloc(const MyAlloc<U, N>& ) {}

	T* allocate(std::size_t sz)
	{
		T* p = mPool.template get<T>();
		printf("allocate %ld cnt, %p\n", sz, p);
		return p;
	}
	void deallocate(T* p, std::size_t sz)
	{
        mPool.release(p);
		printf("deallocate %ld cnt, %p\n", sz, p);
	}
    /*
	template<typename ... ARGS>
	void construct(T* p, ARGS&& ... args)
	{
		printf("call constructor %p\n", p);
		new(p) T(std::forward<ARGS>(args)...);
	}
	void destroy(T* p)
	{
		printf("call destructor %p\n", p);
		p->~T();
	}
    */

    template <typename U>
    struct rebind {
        using other = MyAlloc<U, N>;
    };

private:
    FastMemoryPool<T, N> mPool;
};

template<typename T, int N> 
bool operator==(const MyAlloc<T, N>& a1, const MyAlloc<T, N>& a2)
{
	return true;
}
template<typename T, int N>
bool operator!=(const MyAlloc<T, N>& a1, const MyAlloc<T, N>& a2)
{
	return false;
}
~~~

그리고 queue의 사이즈를 변경해 가며 테스트 해보았습니다.
우선 10개의 메모리 공간만 요청하고 3개의 데이터를 넣어보았습니다.
~~~c++
#include <list>
#include <queue>

int main() {
    std::queue<int, std::list<int, MyAlloc<int, 10>>> q;

    for (auto i = 0; i < 3; i++)
        q.push(i);

    return 0;
}
~~~
아래와 같이 정상적으로 잘 동작합니다.
~~~
allocate 1 cnt, 0x6000015a4000
allocate 1 cnt, 0x6000015a4018
allocate 1 cnt, 0x6000015a4030
deallocate 1 cnt, 0x6000015a4000
deallocate 1 cnt, 0x6000015a4018
deallocate 1 cnt, 0x6000015a4030
~~~

이번엔 13개의 데이터를 넣어 보았습니다. 
~~~
allocate 1 cnt, 0x6000031f4000
allocate 1 cnt, 0x6000031f4018
allocate 1 cnt, 0x6000031f4030
allocate 1 cnt, 0x6000031f4048
allocate 1 cnt, 0x6000031f4060
allocate 1 cnt, 0x6000031f4078
allocate 1 cnt, 0x6000031f4090
allocate 1 cnt, 0x6000031f40a8
allocate 1 cnt, 0x6000031f40c0
allocate 1 cnt, 0x6000031f40d8
allocate 1 cnt, 0x0
[1]    91403 segmentation fault  ./test
~~~
예상대로 segmentation fault가 발생하였습니다. 정해진 사이즈 안에서만 동작하는 것이 명확하면 위와 같이 사용해 볼 수 있습니다.
