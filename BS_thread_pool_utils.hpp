#ifndef BS_THREAD_POOL_UTILS_HPP
#define BS_THREAD_POOL_UTILS_HPP

#include <chrono>           
#include <future>           
#include <initializer_list> 
#include <iostream>         
#include <memory>           
#include <mutex>            
#include <ostream>          
#include <utility>          

namespace BS {

#define BS_THREAD_POOL_UTILS_VERSION_MAJOR 4
#define BS_THREAD_POOL_UTILS_VERSION_MINOR 1
#define BS_THREAD_POOL_UTILS_VERSION_PATCH 0

    class [[nodiscard]] signaller
    {
    public:

        signaller() : promise(), future(promise.get_future()) {}

        signaller(const signaller&) = delete;
        signaller& operator=(const signaller&) = delete;

        signaller(signaller&&) = default;
        signaller& operator=(signaller&&) = default;

        void ready()
        {
            promise.set_value();
        }

        void wait()
        {
            future.wait();
        }

    private:

        std::promise<void> promise;

        std::shared_future<void> future;
    };

    class [[nodiscard]] synced_stream
    {
    public:

        explicit synced_stream(std::ostream& stream = std::cout) : out_stream(stream) {}

        synced_stream(const synced_stream&) = delete;
        synced_stream(synced_stream&&) = delete;
        synced_stream& operator=(const synced_stream&) = delete;
        synced_stream& operator=(synced_stream&&) = delete;

        template <typename... T>
        void print(T&&... items)
        {
            const std::scoped_lock stream_lock(stream_mutex);
            (out_stream << ... << std::forward<T>(items));
        }

        template <typename... T>
        void println(T&&... items)
        {
            print(std::forward<T>(items)..., '\n');
        }

        inline static std::ostream& (&endl)(std::ostream&) = static_cast<std::ostream & (&)(std::ostream&)>(std::endl);

        inline static std::ostream& (&flush)(std::ostream&) = static_cast<std::ostream & (&)(std::ostream&)>(std::flush);

    private:

        std::ostream& out_stream;

        mutable std::mutex stream_mutex = {};
    };

    class [[nodiscard]] timer
    {
    public:

        timer() = default;

        [[nodiscard]] std::chrono::milliseconds::rep current_ms() const
        {
            return (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time)).count();
        }

        void start()
        {
            start_time = std::chrono::steady_clock::now();
        }

        void stop()
        {
            elapsed_time = std::chrono::steady_clock::now() - start_time;
        }

        [[nodiscard]] std::chrono::milliseconds::rep ms() const
        {
            return (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time)).count();
        }

    private:

        std::chrono::time_point<std::chrono::steady_clock> start_time = std::chrono::steady_clock::now();

        std::chrono::duration<double> elapsed_time = std::chrono::duration<double>::zero();
    };
}
#endif