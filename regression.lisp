(defun generate-points-along-function (fn noise-variance n &optional (range 1000))
  "generate n points along a function with some random noise. fn takes x and outputs y"
  (let ((points))
    (dotimes (i n)
      (let* ((x (* (random 1.0) range))
             (y (funcall fn x))
             (noise (* (random 1.0) noise-variance)))
        (push (cons x (+ y noise)) points)))
    points))
(defun polynomial-tensor-activation-function (tensor indicies)
  "decides which degree to apply to a tensor cell based on its indicies"
  (let* ((row-major (apply #'array-row-major-index (append (list tensor) indicies)))
         (myexponent (1+ row-major)) ;; we dont want an exponent of 0
         (value (aref-indicies tensor indicies)))
    (expt value myexponent)))

(defun polynomial-tensor-activation-function-derivative (tensor indicies)
  "the derivative for usage with polynomial-tensor-activation-function"
  (let* ((row-major (apply #'array-row-major-index (append (list tensor) indicies)))
         (original-exponent (1+ row-major))
         (value (aref-indicies tensor indicies)))
    (* original-exponent (expt value (1- original-exponent)))))

(defun make-polynomial-regression-network (&optional (max-degree 3) (learning-rate 0.001))
  (make-network
   :layers (list (make-dense-layer
                  :num-units max-degree
                  :prev-layer-num-units 1
                  :tensor-activation-function #'polynomial-tensor-activation-function
                  :tensor-activation-function-derivative #'polynomial-tensor-activation-function-derivative)
                 (make-dense-layer
                  :num-units 1 :prev-layer-num-units max-degree
                  :activation-function #'identity
                  :activation-function-derivative (lambda (val) 1)))
   :learning-rate learning-rate))

(defun polynomial-regression-test ()
  (let ((nw (make-polynomial-regression-network))
        (data (mapcar (lambda (mycons) (cons (make-array 1 :initial-element (car mycons))
                                             (make-array 1 :initial-element (cdr mycons))))
                      (generate-points-along-function
                       (lambda (x) (+ (expt x 2) 1))
                       2
                       1000
                       1)))
        (epochs 10000))
    (loop repeat epochs do
      (network-train nw data)
      (format t "loss ~A~%" (network-test nw data)))))
