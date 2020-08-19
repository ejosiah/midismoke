#pragma once

namespace ncl {

	template<typename T, size_t N>
	class RingBuffer {
	public:
		inline size_t capacity() const {
			return N;
		}

		inline void push_back(T elm) {
			buffer[++writeSeqeuence % _capacity] = elm;
		}

		T front() const{
			auto index = readSequence % _capacity;
			return buffer[index];
		}

		T pop_front() {
			return buffer[readSequence++ % _capacity];
		}

		inline size_t size() const {
			return writeSeqeuence - readSequence + 1;
		}

		inline bool full() const {
			return size() == _capacity;
		}

		inline bool empty() const {
			return writeSeqeuence < readSequence;
		}

		inline void clear() {
			readSequence = 0;
			writeSeqeuence = -1;
		}

	private:
		int readSequence = 0;
		int writeSeqeuence = -1;
		const size_t _capacity = N;
		T buffer[N];
	};
}