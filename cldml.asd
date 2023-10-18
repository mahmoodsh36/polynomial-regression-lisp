(defsystem
 "cldml"
 :description "machine learning in common lisp"
 :version "0.0.1"
 :serial t
 :license "BSD"
 :pathname "."
 :depends-on ("serapeum" "lparallel" "hunchentoot" "jsown")
 :components ((:file "network") (:file "utils") (:file "math-utils") (:file "regression")))
