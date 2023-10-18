;; (defun array-width (2d-array)
;;   (car (cdr (array-dimensions 2d-array))))
;; (defun array-height (2d-array)
;;   (car (array-dimensions 2d-array)))
(defun array-rows (arr)
  "number of rows in array"
  (elt (array-dimensions arr) (- (length (array-dimensions arr)) 2)))
(defun array-cols (arr)
  "number of columns in array"
  (elt (array-dimensions arr) (- (length (array-dimensions arr)) 1)))
(defun array-depth (arr)
  "depth of a tensor of the third order"
  (elt (array-dimensions arr) (- (length (array-dimensions arr)) 3)))
(defun array-size (arr)
  (array-total-size arr))
(defun list-dimensions (list)
  (if (and (eq (type-of list) 'cons) (eq (type-of (car list)) 'cons))
      (append (list (length list)) (list-dimensions (car list)))
      (list (length list))))
(defun list->array (list)
  (make-array (list-dimensions list)
              :initial-contents list))
(defun random-float (min max)
  (+ min (* (random 1.0) (- max min))))
(defun random-list (dims &optional (min -1) (max 1))
  (let* ((dim (car dims)))
    (let ((list (make-list dim)))
      (if (> (length dims) 1)
          (loop for i from 0 below dim
                do (setf (elt list i) (random-list (cdr dims))))
          (loop for i from 0 below dim
                do (setf (elt list i) (random-float min max))))
      list)))
(defun random-tensor (dims &optional (min -1) (max 1))
  "an inefficient way, but ill take it for now."
  (if (atom dims)
      (list->array (random-list (list dims) min max))
      (list->array (random-list dims min max))))
(defun array-map (fn &rest arrays)
  "apply fn to elements of arrays. return a new array with the results"
  (apply #'array-map-into
         (append (list fn (make-array (array-dimensions (car arrays))))
                 arrays)))
(defun array-map-into (fn dest-array &rest arrays)
  "call fn with elements of arrays. write results into dest-array"
  (dotimes (i (array-total-size (car arrays)) dest-array)
    (setf (row-major-aref dest-array i)
          (apply fn (loop for array in arrays collect (row-major-aref array i))))))
(defun array-map-indicies-into (array fn dest-array)
  "apply fn to each combination of indicies of array, return a new array with the results, the argument to fn should be the indicies list, e.g. (3 5 3) to be used with aref, e.g. (aref arr 3 5 3)"
  (dotimes (i (array-total-size array) dest-array)
    (setf (row-major-aref dest-array i)
          (apply fn (list array (from-row-major (array-dimensions dest-array) i))))))
(defun array-map-indicies (array fn)
  "see description of array-map-by-idx-into, this function is an alias for it that provides a new array as dest-array"
  (array-map-indicies-into array fn (make-array (array-dimensions array))))
(defun aref-indicies (arr idx-list)
  "to shorten code, not having to use apply everywhere"
  (apply #'aref (cons arr idx-list)))
(defun subarray (a offsets dims)
  "copy a subarray of a given array
example usage:
(subarray #2A((0.70462835 0.7893367 0.3833828)
              (0.8126608 0.6136594 0.37239313)
              (0.44657052 0.4761132 0.9504193)) '(1 0) '(2 2))"
  (let* ((b (make-array dims)))
    (subarray-util a b nil nil offsets (array-dimensions a) dims)
    b))

(defun subarray-util (a b a-indicies b-indicies offsets a-dims b-dims)
  (if b-dims
      (let ((use-b-dims (eq (length a-dims) (length b-dims)))
            (offset (car offsets)))
        (if use-b-dims
            (loop for i from offset below (+ offset (car b-dims))
                  do (subarray-util
                      a
                      b
                      (append a-indicies (list i))
                      (append b-indicies (list (- i offset)))
                      (cdr offsets)
                      (cdr a-dims)
                      (cdr b-dims)))
            (subarray-util
             a
             b
             (append a-indicies (list offset))
             nil
             (cdr offsets)
             (cdr a-dims)
             b-dims)))
      (setf (apply #'aref (cons b b-indicies))
            (apply #'aref (cons a a-indicies)))))
(defun copy-array-nth (arr idx)
  "return a copy of the nth subarray of an array"
  (let* ((new-dims (cdr (array-dimensions arr)))
         (offsets (append (list idx) (make-list (length new-dims) :initial-element 0))))
    (subarray
     arr
     offsets
     new-dims)))
(defun array-nth (arr idx)
  "return the nth subarray of an array without copying"
  (let ((new-dims (cdr (array-dimensions arr))))
    (make-array new-dims
                :displaced-to arr
                :displaced-index-offset (* idx (reduce #'* new-dims)))))
(defun copy-into-array (a b &optional (offsets (make-list (length (array-dimensions a)) :initial-element 0)))
  "copy an (possibly smaller) array (b) into another array (a)"
  (copy-into-array-util a b nil nil offsets (array-dimensions a) (array-dimensions b))
  a)

(defun copy-into-array-util (a b a-indicies b-indicies offsets a-dims b-dims)
  (if b-dims
      (let ((use-b-dims (eq (length a-dims) (length b-dims)))
            (offset (car offsets)))
        (if use-b-dims
            (loop for i from offset below (+ offset (car b-dims))
                  do (copy-into-array-util
                      a
                      b
                      (append a-indicies (list i))
                      (append b-indicies (list (- i offset)))
                      (cdr offsets)
                      (cdr a-dims)
                      (cdr b-dims)))
            (copy-into-array-util
             a
             b
             (append a-indicies (list offset))
             nil
             (cdr offsets)
             (cdr a-dims)
             b-dims)))
      (setf (apply #'aref (cons a a-indicies))
            (apply #'aref (cons b b-indicies)))))
(defun set-array-nth (arr other-arr idx)
  "example usage: (set-array-nth (make-array '(4 4 4)) (random-tensor '(4 4)) 1)"
  (copy-into-array
   arr
   other-arr
   (append (list idx) (make-list (length (array-dimensions other-arr)) :initial-element 0))))
(defun vectorize-array (arr)
  "turn an array into a 1d vector"
  (make-array (list (array-total-size arr)) :displaced-to arr))
(defun array-index-row-major (array rmi)
  "basically the inverse of array-row-major-index, example:
(defvar *a* (make-array '(7 4 9 5)))
(array-row-major-index *a* 3 2 8 1) => 671
(array-index-row-major *a* 671) => (3 2 8 1)
"
  (reduce #'(lambda (dim x)
              (nconc
               (multiple-value-list (truncate (car x) dim))
               (cdr x)))
          (cdr (array-dimensions array))
          :initial-value (list rmi)
          :from-end t))

(defun from-row-major (dims row-major-idx)
  "same function as array-index-row-major, but not specific to arrays"
  (reduce #'(lambda (dim x)
              (nconc
               (multiple-value-list (truncate (car x) dim))
               (cdr x)))
          (cdr dims)
          :initial-value (list row-major-idx)
          :from-end t))
(defun copy-array (arr)
  (let ((new-arr (make-array (array-dimensions arr))))
    (copy-into-array new-arr arr)))
(defun reduce-array (arr fn)
  (apply #'reduce
         (list fn (make-array (array-total-size arr) :displaced-to arr))))
(defun last-elt (seq)
  "example usage: CL-USER> (last-elt (vector 1 2 3)) => 3"
  (elt seq (1- (length seq))))
(defun list->vector (list)
  (if (atom list) list
      (map 'vector #'list->vector list)))
;; (defun argmax (s &key (key #'identity))
;;   "returns value,index of element in sequence s that maximizes the key function"
;;   (when s
;;     (let* ((m (reduce #'max s :key key))
;;            (idx (position m s :key key)))
;;       (values (elt s idx) idx))))
;; 
;; (defun argmin (s &key (key #'identity))
;;   "returns value,index of element in sequence s that minimizes the key function"
;;   (when s
;;     (let* ((m (reduce #'min s :key key))
;;            (idx (position m s :key key)))
;;       (values (elt s idx) idx))))
;; from https://github.com/eudoxia0/trivial-open-browser/blob/master/src/trivial-open-browser.lisp

(defparameter +format-string+
              #+(or win32 mswindows windows)
              "explorer ~S"
              #+(or macos darwin)
              "open ~S"
              #-(or win32 mswindows macos darwin windows)
              "xdg-open ~S")

(defun open-browser-through-shell (url)
  "run a shell command to open `url`."
  (uiop:run-program (format nil +format-string+ url)))

(defparameter
 *browser-function*
 #'open-browser-through-shell
 "the function that gets called with the URL to open the browser. defaults to
  `#'open-browser-through-shell`.")

(defun open-browser (url)
  "open the browser to `url`."
  (funcall *browser-function* url))
