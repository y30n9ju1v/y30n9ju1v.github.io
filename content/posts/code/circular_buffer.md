+++
title = 'Circular Buffer'
date = 2024-07-27T08:53:08+09:00
draft = false
+++

지난번에 작성한 [FastMemoryPool]({{< ref "fast_memory_pool" >}}) 을 사용하여 센서에서 들어오는 데이터를 버퍼링하는 클래스를 작성하였습니다. 
1. 버퍼는 입력된 순서대로만 접근합니다.
2. 버퍼 사이즈를 크게 하여 못 가져간 예전 메모리는 재사용합니다.

~~~c++
#include "fastmemorypool.h"

#include <queue>

template <typename T, int N = 4>
class CircularBuffer final
{
public:
    T* front()
    {
        if (memoryQueue.size() == N) pop();
        T* ptr = memoryPool.template get<T>();
        memoryQueue.push(ptr);
        return ptr;
    }

    int available() const {
        return N - memoryQueue.size();
    }

protected:
    void pop()
    {
        if (memoryQueue.empty()) return;
        memoryPool.release(memoryQueue.front());
        memoryQueue.pop();
    }

private:
    std::queue<T*> memoryQueue;
    FastMemoryPool<T, N> memoryPool;
};
~~~
