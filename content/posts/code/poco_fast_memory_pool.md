+++
title = 'Fast Memory Pool in POCO'
date = 2024-07-26T21:54:22+09:00
draft = false
+++

이 글은 https://github.com/pocoproject/poco/blob/main/Foundation/include/Poco/MemoryPool.h 의 FastMemoryPool을 설명한 글입니다.

FastMemoryPool은 고정 크기 메모리 블록을 풀링하는 클래스입니다.
이 클래스의 주요 목적은 메모리 할당 속도를 높이고, 서버 애플리케이션과 같이 동일한 블록이 반복적으로 할당되는 상황에서 메모리 단편화를 줄이는 것입니다.
이 클래스는 블록 크기를 결정하는 방식에서 MemoryPool과 다릅니다.
블록 크기는 보유한 타입 크기에서 추론되어 정적으로 적용됩니다.
이름에서 알 수 있듯이 Poco::MemoryPool보다 빠릅니다.
또한, 런타임 플랫폼의 일반적인 메모리 할당 기능보다도 상당히 빠를 가능성이 있지만, 일정한 제한 사항이 있습니다(고정 크기 블록만 제공하는 것 외에도)
FastMemoryPool은 포인터보다 작은 타입의 배열에는 사용할 수 없습니다.

간단한 사용 예 입니다.

풀에서 메모리를 사용하는 객체는 인플레이스 new 연산자를 사용하여 생성해야 합니다.
객체가 풀에 반환되면, 풀에 의해 해당 객체의 소멸자가 호출됩니다.
반환된 포인터는 메모리를 얻었던 타입에 대한 유효한 포인터여야 합니다.

~~~c++
Example use:

  using std::vector;
  using std:string;
  using std::to_string;
  using Poco::FastMemoryPool;

  int blocks = 10;
  FastMemoryPool<int> fastIntPool(blocks);
  FastMemoryPool<string> fastStringPool(blocks);

  vector<int*> intVec(blocks, 0);
  vector<string*> strVec(blocks);

  for (int i = 0; i < blocks; ++i)
  {
    intVec[i] = new (fastIntPool.get()) int(i);
    strVec[i] = new (fastStringPool.get()) string(to_string(i));
  }

  for (int i = 0; i < blocks; ++i)
  {
    fastIntPool.release(intVec[i]);
    fastStringPool.release(strVec[i]);
  }
~~~

풀은 메모리 블록을 “버킷”에 저장합니다.
버킷은 블록의 배열로, 항상 단일 new[]로 할당되며, 생성 시 블록이 초기화됩니다.
풀의 현재 용량이 초과될 때마다 새로운 버킷이 할당되고 그 블록들은 내부적으로 사용되도록 초기화됩니다.
새로운 버킷 할당이 허용된 최대 크기를 초과할 경우, std::bad_alloc() 예외가 발생하며 객체 자체는 손상되지 않습니다.

FastMemoryPool은 스레드 안전합니다.
기본적으로 Poco::FastMutex를 사용하지만, 필요한 경우 템플릿 매개변수를 통해 다른 뮤텍스를 지정할 수 있습니다.
단일 스레드 시나리오에서 잠금을 피하고 속도를 높이기 위해 템플릿 매개변수로 Poco::NullMutex를 지정할 수 있습니다.

~~~c++
//
// FastMemoryPool
//

#define POCO_FAST_MEMORY_POOL_PREALLOC 1000

template <typename T, typename M = FastMutex>
class FastMemoryPool
{
private:
~~~

아래 클래스는 메모리 블록을 나타냅니다.
이 클래스는 두 가지 용도를 가지는데, 주요 용도는 풀의 사용자에게 제공되는 메모리를 관리하는 것입니다.
두 번째 용도는 내부 “관리” 목적을 위한 것입니다.

작동 방식은 다음과 같습니다:
* 처음 생성될 때, 블록은 적절하게 생성되고 내부 블록 연결 리스트에 위치합니다.
* 사용자가 사용할 때, 블록은 내부 블록 연결 리스트에서 제거됩니다.
* 풀에 반환될 때, 블록은 다시 인플레이스 생성되어 내부 블록 연결 리스트의 다음 사용 가능한 블록으로 삽입됩니다.

블록을 생성하고 다음 포인터를 설정합니다.
이 생성자는 새로 할당된 버킷에서 블록 시퀀스(블록 배열)를 초기화하는 데만 사용해야 합니다.
~~~c++
	class Block
	{
	public:
		Block()
		{
			_memory.next = this + 1;
		}

		explicit Block(Block* next)  // Creates a Block and sets its next pointer.
		{
			_memory.next = next;
		}

#ifndef POCO_DOC
		union
		{
			char buffer[sizeof(T)];
			Block* next;
		} _memory;
#endif

	private:
		Block(const Block&);
		Block& operator = (const Block&);
	};

public:
	FastMemoryPool(std::size_t blocksPerBucket = POCO_FAST_MEMORY_POOL_PREALLOC,
            std::size_t bucketPreAlloc = 10, std::size_t maxAlloc = 0):
			_blocksPerBucket(blocksPerBucket),
			_maxAlloc(maxAlloc),
			_available(0)
	{
		if (_blocksPerBucket < 2)
			throw std::invalid_argument("FastMemoryPool: blocksPerBucket must be >=2");
		_buckets.reserve(bucketPreAlloc);
		resize();
	}

	~FastMemoryPool()
	{
		clear();
	}
~~~

다음 사용 가능한 메모리 블록에 대한 포인터를 반환합니다.
풀이 소진되면 새로운 버킷을 할당하여 크기가 조정됩니다.    
~~~c++
	void* get()
	{
		Block* ret;
		{
			M::ScopedLock l(_mutex);
			if(_firstBlock == 0) resize();
			ret = _firstBlock;
			_firstBlock = _firstBlock->_memory.next;
		}
		--_available;
		return ret;
	}
~~~
반환된 메모리를 내부 용도로 초기화하고 다음 사용 가능한 블록으로 설정하여 재활용합니다.
이전의 다음 블록은 이 블록의 다음 블록이 됩니다.
null 포인터의 반환은 조용히 무시됩니다.
반환된 포인터에 대한 소멸자가 호출됩니다.
~~~c++
	template <typename P>
	void release(P* ptr)
	{
		if (!ptr) return;
		reinterpret_cast<P*>(ptr)->~P();
		++_available;
		M::ScopedLock l(_mutex);
		_firstBlock = new (ptr) Block(_firstBlock);
	}

    // Returns the block size in bytes.
	std::size_t blockSize() const  
	{
		return sizeof(Block);
	}

    // Returns the total amount of memory allocated, in bytes.
	std::size_t allocated() const  
	{
		return _buckets.size() * _blocksPerBucket;
	}

    // Returns currently available amount of memory in bytes.
	std::size_t available() const  
	{
		return _available;
	}

private:
	FastMemoryPool(const FastMemoryPool&);
	FastMemoryPool& operator = (const FastMemoryPool&);
~~~
새 버킷을 생성하고 내부 용도로 초기화합니다.
이전의 다음 블록을 새 버킷의 첫 번째 블록을 가리키도록 설정하고, 새 버킷의 마지막 블록이 마지막 블록이 됩니다.
~~~c++
	void resize()
	{
		if (_buckets.size() == _buckets.capacity())
		{
			std::size_t newSize = _buckets.capacity() * 2;
			if (_maxAlloc != 0 && newSize > _maxAlloc) throw std::bad_alloc();
			_buckets.reserve(newSize);
		}
		_buckets.push_back(new Block[_blocksPerBucket]);
		_firstBlock = _buckets.back();
		// terminate last block
		_firstBlock[_blocksPerBucket-1]._memory.next = 0;
		_available = _available.value()
            + static_cast<AtomicCounter::ValueType>(_blocksPerBucket);
	}

	void clear()
	{
		typename std::vector<Block*>::iterator it = _buckets.begin();
		typename std::vector<Block*>::iterator end = _buckets.end();
		for (; it != end; ++it) delete[] *it;
	}

	const std::size_t _blocksPerBucket;
	std::vector<Block*> _buckets;
	Block* _firstBlock;
	std::size_t _maxAlloc;
	Poco::AtomicCounter _available;
	mutable M _mutex;
};
~~~

위 FastMemoryPool을 resize없이 정해진 사이즈의 메모리 풀을 가지는 클래스로 바꾼 코드입니다.

~~~c++
#include <atomic>
#include <mutex>

template <typename T, int N = 4>
class FastMemoryPool
{
private:
	class Block
	{
	public:
		Block() { _memory.next = this + 1; }
		explicit Block(Block* next) { _memory.next = next; }

		union {
			char buffer[sizeof(T)];
			Block* next;
		} _memory;

	private:
		Block(const Block&);
		Block& operator = (const Block&);
	};

public:
	FastMemoryPool() {
        static_assert(sizeof(T) >= sizeof(void*),
                "T must be same or larger than pointer size");

        _blocks = new Block[N];
        _blocks[N - 1]._memory.next = nullptr;
        _firstBlock = _blocks;
	}

	~FastMemoryPool() { delete[] _blocks; }

	template <typename P>
	P* get() {
		Block* ret;
		{
			std::scoped_lock<std::mutex> l(_mutex);
			if(_firstBlock == nullptr) return nullptr;

			ret = _firstBlock;
			_firstBlock = _firstBlock->_memory.next;
		}
		--_available;
		return reinterpret_cast<P*>(ret);
	}

	template <typename P>
	void release(P* ptr) {
		if (!ptr) return;
		reinterpret_cast<P*>(ptr)->~P();
		++_available;
		std::scoped_lock<std::mutex> l(_mutex);
		_firstBlock = new (ptr) Block(_firstBlock);
	}

	int available() const { return _available; }

private:
	FastMemoryPool(const FastMemoryPool&);
	FastMemoryPool& operator = (const FastMemoryPool&);

	Block* _blocks{nullptr};
	Block* _firstBlock{nullptr};
    std::atomic<int> _available{N};
	mutable std::mutex _mutex;
};
~~~
