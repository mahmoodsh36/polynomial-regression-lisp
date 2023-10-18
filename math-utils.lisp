(defun sigmoid (x)
  (cond ((< x -70) 0) ;; to avoid float overflow
        (t (/ 1 (1+ (exp (- x)))))))

(defun sigmoid-derivative (x)
  (* (sigmoid x) (- 1 (sigmoid x))))

(defun relu (x)
  (max x 0))

(defun relu-derivative (x)
  "derivative of relu is undefined at x=0, but here we return 0 for x=0 too"
  (if (> x 0)
      1
      0))
;; ;; this might need efficiency improvement, not sure
;; (defun vector-add (vec1 &rest more-vecs)
;;   "apply vector addition to two collections"
;;   (apply #'map (append (list 'vector #'+) (append (list vec1) more-vecs))))
;; 
;; ;; this definitely needs efficiency improvement, we can just subtract instead of negating vectors then adding them
;; (defun vector-sub (vec1 &rest more-vecs)
;;   "substract each vector in more-vecs from vec1"
;;   (apply #'vector-add
;;          (append (list vec1)
;;                  (map 'list
;;                       (lambda (vec2)
;;                         (map 'vector #'- vec2))
;;                       more-vecs))))
;; 
;; (defun vector-dot (vec1 vec2)
;;   "dot product"
;;   (reduce #'+ (map 'vector #'* vec1 vec2)))
;; 
;; (defun vector-mul (vec1 scalar)
;;   "multiply vectors elements by scalar"
;;   (map 'vector (lambda (num) (* num scalar)) vec1))
;; 
;; (defun vector-sum (vec1 &rest more-vecs)
;;   "sum of vectors elements"
;;   (reduce #'+ (map
;;                'vector
;;                (lambda (vec2) (reduce #'+ vec2))
;;                (append (list vec1) more-vecs))))
;; 
;; (defun vector-elements-prod (vec)
;;   (reduce #'* vec))
;; (defun matrix-mul (arr1 arr2)
;;   "multiply 2 matrices, arr1*arr2"
;;   (let* ((arr1-rows (array-rows arr1))
;;          ;; (arr1-cols (car (cdr arr1-dim))) ;; no need
;;          (arr2-rows (array-rows arr2))
;;          (arr2-cols (array-cols arr2))
;;          (out-arr (make-array (list arr1-rows arr2-cols))))
;;     (loop for arr1-row from 0 below arr1-rows
;;           do (loop for arr2-col from 0 below arr2-cols
;;                    do (let ((sum 0))
;;                         (loop for i from 0 below arr2-rows
;;                               do (let* ((cell1 (aref arr1 arr1-row i))
;;                                         (cell2 (aref arr2 i arr2-col)))
;;                                    (incf sum (* cell1 cell2))))
;;                         (setf (aref out-arr arr1-row arr2-col) sum))))
;;     out-arr))
(defun tensor-convolution-size (img-dims ker-dims)
  "return the expected size of the convolution img*ker"
  (mapcar (lambda (img-d ker-d) (- img-d (1- ker-d)))
          img-dims
          ker-dims))

(defun tensor-convolution (img ker)
  "convolve two tensors of any ranks/dimensions
example usage: (tensor-convolution #(1 2 3 4 5) #(6 7))"
  (let ((ker-dims (array-dimensions ker))
        (img-dims (array-dimensions img)))
    ;; need to make ker-dims,out-dims same rank/order as img-dims
    (loop while (< (length ker-dims) (length img-dims))
          do (push 1 ker-dims)
             (setf ker (make-array ker-dims :displaced-to ker)))
    (let ((out (make-array (tensor-convolution-size img-dims ker-dims)
                           :initial-element 0)))
      (tensor-convolution-util img ker out nil nil (array-dimensions out))
      out)))

(defun tensor-convolution-util (img ker out
                                img-indicies out-indicies
                                out-dims)
  "this util function for tensor-convolution assumes img,ker,out are all of the same rank/order"
  (if out-dims
      ;; iterate through each cell in the output tensor
      (loop for out-i from 0 below (car out-dims)
            ;; iterate through each kernel-sized block in the input tensor
            ;; (length img-indicies) gives us the index of the dimension we're at
            do (loop for img-i from out-i below (+ out-i
                                                   (array-dimension
                                                    ker
                                                    (length img-indicies)))
                     do (tensor-convolution-util
                         img
                         ker
                         out
                         (append img-indicies (list img-i))
                         (append out-indicies (list out-i))
                         (cdr out-dims))))
      (let* ((img-cell (apply #'aref (cons img img-indicies)))
             (flipped-kernel-indicies (mapcar #'-
                                              (mapcar #'1- (array-dimensions ker))
                                              (mapcar #'- img-indicies out-indicies)))
             (ker-cell (apply #'aref (cons ker flipped-kernel-indicies))))
        (incf (apply #'aref (cons out out-indicies)) (* img-cell ker-cell)))))
;; (defun map-grid-matrix (src-x-range dest-x-range src-y-range dest-y-range)
;;   (let ((src-x-min (elt src-x-range 0))
;;         (dest-x-min (elt dest-x-range 0))
;;         (src-y-min (elt src-y-range 0))
;;         (dest-y-min (elt dest-y-range 0))
;;         (src-x-max (elt src-x-range 1))
;;         (dest-x-max (elt dest-x-range 1))
;;         (src-y-max (elt src-y-range 1))
;;         (dest-y-max (elt dest-y-range 1)))
;;     (make-array
;;      '(3 3)
;;      :initial-contents
;;      `((,(/ (- dest-x-max dest-x-min) (- src-x-max src-x-min)) 0 ,(/ (+ (* (- src-x-min) dest-x-max) (* dest-x-min src-x-max)) (- src-x-max src-x-min)))
;;        (0 ,(/ (- dest-y-max dest-y-min) (- src-y-max src-y-min)) ,(/ (+ (* (- src-y-min) dest-y-max) (* dest-y-min src-y-max)) (- src-y-max src-y-min)))
;;        (0 0 1)))))
;; (defun map-grid (mat src-x-range dest-x-range src-y-range dest-y-range)
;;   "apply the grid mapping matrix and return a new matrix"
;;   (matrix-mul
;;    (map-grid-matrix src-x-range dest-x-range src-y-range dest-y-range)
;;    mat))
;; (defun map-num (num src-min src-max dest-min dest-max)
;;   "(map-num 0.5 -1 1 -50 50) => 25.0"
;;   (/ (- (+ (* (- num src-min) (- dest-max dest-min)) (* dest-min src-max)) (* dest-min src-min)) (- src-max src-min)))
