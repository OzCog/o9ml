# Minimal logger declarations for URE module
from libcpp cimport bool
from libcpp.string cimport string

cdef extern from "opencog/util/Logger.h" namespace "opencog":
    enum loglevel "opencog::Logger::Level":
        NONE "opencog::Logger::NONE"
        ERROR "opencog::Logger::ERROR"
        WARN "opencog::Logger::WARN"
        INFO "opencog::Logger::INFO"
        DEBUG "opencog::Logger::DEBUG"
        FINE "opencog::Logger::FINE"
        BAD_LEVEL "opencog::Logger::BAD_LEVEL"
    
    cdef cppclass cLogger "opencog::Logger":
        cLogger()
        cLogger(string s)
        void set_level(loglevel lvl)
        void set_component(string c)
        loglevel get_level()
        void set_print_to_stdout_flag(bool flag)
        void set_sync_flag(bool flag)
        void log(loglevel lvl, string txt)
        bool is_enabled(loglevel lvl)
    
    cdef loglevel string_to_log_level "opencog::Logger::get_level_from_string"(string s)
    cdef string log_level_to_string "opencog::Logger::get_level_string"(loglevel lvl)
    cLogger& logger()