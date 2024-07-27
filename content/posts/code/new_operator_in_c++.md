+++
title = 'New Operator in C++'
date = 2024-07-27T10:07:29+09:00
draft = false
+++

문득 c++에서 operator new[]의 동작이 궁금해 졌습니다.
생성자를 가진 객체의 new[]로 동적할당하면 어떻게 될까? 설마 하나 메모리 할당받고 생성자 호출하진 않겠지만 코드로 확인해 보기로 했습니다.
코드는 https://en.cppreference.com/w/cpp/memory/new/operator_new 의 코드를 조금 수정하였습니다.

~~~c++
#include <cstdio>
#include <cstdlib>
#include <new>
 
void* operator new(std::size_t sz) {
    std::printf("1) new(size_t), size = %zu\n", sz);
    if (sz == 0) ++sz; // avoid std::malloc(0) which may return nullptr on success
 
    if (void *ptr = std::malloc(sz)) {
        printf("1) new(size_t), ptr = %d\n", ptr);
        return ptr;
    }
 
    throw std::bad_alloc{};
}
 
void* operator new[](std::size_t sz) {
    std::printf("2) new[](size_t), size = %zu\n", sz);
    if (sz == 0) ++sz; // avoid std::malloc(0) which may return nullptr on success
 
    if (void *ptr = std::malloc(sz)) {
        printf("2) new[](size_t), ptr = %d\n", ptr);
        return ptr;
    }
 
    throw std::bad_alloc{};
}
 
void operator delete(void* ptr) noexcept {
    std::puts("3) delete(void*)");
    std::free(ptr);
}

void operator delete[](void* ptr) noexcept {
    std::puts("5) delete[](void* ptr)");
    std::free(ptr);
}

class myClass {
public:
    myClass() : tmp(this) {
        printf("Construct(%d)\n", this);
    }
    ~myClass() {
        printf("Deconstruct(%d)\n", this);
    }
private:
    myClass* tmp;
};

int main() {
    int* p1 = new int;
    delete p1;
 
    int* p2 = new int[4];
    delete[] p2;

    auto* p3 = new myClass;
    delete p3;

    auto* p4 = new myClass[4]();
    delete[] p4;
}
~~~

결과는 예상한 대로 다음과 같습니다. 한번에 크게 메모리를 할당하고 순차적으로 생성자를 호출했습니다.
~~~
1) new(size_t), size = 4
1) new(size_t), ptr = 24526912
3) delete(void*)
2) new[](size_t), size = 16
2) new[](size_t), ptr = 24526912
5) delete[](void* ptr)
1) new(size_t), size = 8
1) new(size_t), ptr = 24526912
Construct(24526912)
Deconstruct(24526912)
3) delete(void*)
2) new[](size_t), size = 48
2) new[](size_t), ptr = 28705488
Construct(28705504)
Construct(28705512)
Construct(28705520)
Construct(28705528)
Deconstruct(28705528)
Deconstruct(28705520)
Deconstruct(28705512)
Deconstruct(28705504)
5) delete[](void* ptr)
~~~

그런데 한가지 신기한점이 있었습니다.
new[]을 사용하면 추가 여분의 메모리가 할당된다는 것입니다. 
저는 8바이트짜리 4개를 할당 요청했는데 32바이트가 아닌 48바이트가 할당되었습니다.
28705504 - 28705488 = 16 byte로 64bit 환경에서 정수형 2개 또는 포인터 2개의 사이즈입니다.
디버거를 이용해 확인해 보았습니다. 

~~~
* thread #1, queue = 'com.apple.main-thread', stop reason = step over
    frame #0: 0x0000000100003bc0 test`main at test.cpp:63:14
   60  	    delete p3;
   61
   62  	    auto* p4 = new myClass[4]();
-> 63  	    delete[] p4;
   64  	}
(lldb) memory read --size 8 --format x --count 8 0x0000600000514000
0x600000514000: 0x0000000000000008 0x0000000000000004
0x600000514010: 0x0000600000514010 0x0000600000514018
0x600000514020: 0x0000600000514020 0x0000600000514028
0x600000514030: 0x0000000000000000 0x0000000000000000
~~~

정확한 것은 컴파일러의 구현된 코드를 봐야 겠지만 앞에 것은 사이즈, 뒤에 것은 개수를 나타내는 것처럼 보입니다.
제가 사용한 컴파일의 정보는 다음과 같습니다.
Apple clang version 15.0.0 (clang-1500.3.9.4)
Target: arm64-apple-darwin23.5.0
