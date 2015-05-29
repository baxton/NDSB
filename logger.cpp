
/*
 * Logger class implementation (for Abhi)
 *
 * author: Maxim Alekseykin
 *
 * Requirements:
 *  1) Output should be done asynchromously
 *  2) Logging should not block working threads
 *  3) Should be as fast as possible
 *
 *
 */




#include <thread>
#include <atomic>
#include <memory>
#include <chrono>
#include <fstream>
#include <string>


/*
 * This namespace contains utility classes to work with memory
 *
 *
 */
namespace memory {


/*
 * Simple allocator for strings
 * In this implementation allocates data on the heap
 * But for performance purpose it can be replaced by "Small objects allocator"
 */
template<class T>
class allocator {
public:

    // the class should not be instanciated or copied
    allocator() = delete;
    allocator(const allocator&) = delete;
    ~allocator() = delete;
    allocator& operator=(const allocator&) = delete;

    // as soon as it's a general template and I do not use it in this test
    // I'm not going to implement these methods here
    static T* allocate();
    static void release(T* p);
};

/*
 * specialization for C strings
 */
template<>
class allocator<char> {
public:

    allocator() = delete;
    allocator(const allocator&) = delete;
    ~allocator() = delete;
    allocator& operator=(const allocator&) = delete;

    static char* allocate(size_t size) {
        return new char[size];
    }

    static void release(const char* p) {
        delete [] p;
    }

};


}


/*
 * This namespace contains everything I need for implementing lock-free
 * request queing
 */
namespace lock_free {

template<class T>
struct node {
    T* val     = nullptr;
    node* next = nullptr;
};


/*
 * Simple linked-list based queue
 * lock free for pushing new items
 */
template<class T>
class que {
    std::atomic<node<T>*> head = {nullptr};

public:
    que() = default;
    que(const que&) = delete;
    que& operator=(const que&) = delete;

    void push(T* pVal) {
        auto p = new node<T>();
        p->val = pVal;
        p->next = head.load();

        while (!head.compare_exchange_weak(p->next, p))
        {/* some logic can be here to stop the loop if necessary */}
    }

    /* this method refreshes the internal que to empty one
     * and return already filled one for further processing
     * clients are able to continue filling in the new empty que
     */
    node<T>* dump_que() {
        auto p = head.load();
        while (!head.compare_exchange_weak(p, nullptr)) {}
        return p;
    }
};


}





/*
 *
 *
 */

namespace log {

enum Severity {
    Trace,
    Debug,
    Info,
    Error,
};

template<class ALLOCATOR = memory::allocator<char>>
class Logger {

    // my lock-free queue for messages
    lock_free::que<char> que;

    // atomic flag to shutdown the working thread
    std::atomic<bool> shutdown = {false};

    Severity current_level = Debug;
    int current_timeout    = 2;        // in seconds

    //std::thread worker(Logger::get_it_done, this);

    // atomic pointer to the output stream allows me to change output file from any thread
    // I use shared pointer for output stream object to bypass ABA problem with lock-free logic
    std::atomic<std::shared_ptr<std::ofstream>> log_stream;

    void get_it_done(Logger* This) {

        while (false == This->shutdown.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(current_timeout));

            // get what ever we've already got in the buffer
            lock_free::node<char>* p = This->que.dump_que();

            if (p) {

            }
        }
    }

public:
    Logger() = default;
    Logger(int timeout, Severity level) :
        current_timeout(timeout),
        current_level(level)
    {}
    Logger(const Logger&) = delete;
    Logger& operator= (const Logger&) = delete;

    ~Logger() {}

    void SetLogFile(const std::string& fname) {

        //std::ofstream* fout =

    }

    void StartBackGroundThread() {
    }

    void ExitLogger() {
    }


    /*
     * The message "msg" must be already prepared by the caller
     * The only requirement here is: I should be able to relese "msg"
     * by the allocator used by the instance of the Logger class
     */
    void Log(Severity level, const char* msg) {
        if (level >= current_level)
            que.push(msg);
    }

};

}





/*
 * Tests
 *
 */




int main(int argc, const char* argv[]) {

    log::Logger<memory::allocator<char>> logger;

    return 0;
}
