# distutils: language = c++
# cython: language_level=3
from cython.operator cimport dereference as deref
from atomspace cimport Atom, AtomSpace, cHandle, cAtomSpace
from opencog.atomspace import types
from ure cimport cBackwardChainer

# Create a Cython extension type which holds a C++ instance
# as an attribute and create a bunch of forwarding methods
# Python extension type.


cdef class BackwardChainer:
    cdef cBackwardChainer * chainer
    cdef AtomSpace _as
    cdef AtomSpace _trace_as
    cdef AtomSpace _control_as
# scheme interface
#    (define* (cog-bc rbs target
#                 #:key
#                 (vardecl (List)) (trace-as #f) (control-as #f) (focus-set (Set)))
    def __cinit__(self, AtomSpace _as,
                  Atom rbs,
                  Atom target,
                  Atom vardecl=None,
                  AtomSpace trace_as=None,
                  AtomSpace control_as=None,
                  Atom focus_set=None):
        cdef cHandle c_vardecl
        if vardecl is None:
            c_vardecl = cHandle.UNDEFINED
        else:
            c_vardecl = deref(vardecl.handle)
        if focus_set is None:
            focus_set = _as.add_link(types.SetLink, [])
        cdef cHandle rbs_handle = deref(rbs.handle)
        cdef cHandle target_handle = deref(target.handle)
        cdef cHandle focus_set_handle = deref(focus_set.handle)
        self.chainer = new cBackwardChainer(deref(_as.atomspace),
                                        <const cHandle&>rbs_handle,
                                        <const cHandle&>target_handle,
                                        <const cHandle&>c_vardecl,
                                        <cAtomSpace*> (NULL if trace_as is None else trace_as.atomspace),
                                        <cAtomSpace*> (NULL if control_as is None else control_as.atomspace),
                                        <const cHandle&>focus_set_handle)
        self._as = _as
        self._trace_as = trace_as
        self._control_as = control_as

    def do_chain(self):
        return self.chainer.do_chain()

    def get_results(self):
        cdef cHandle res_handle = self.chainer.get_results()
        cdef Atom result = Atom.createAtom(res_handle)
        return result

    def __dealloc__(self):
        del self.chainer
        self._trace_as = None
        self._control_as = None
        self._as = None
