(defgeneric feedforward (layer x)
  (:documentation "feed-forward, return activations to pass to next layer"))

(defgeneric propbackward (layer layer-x layer-y layer-y-unactivated propped-deltas learning-rate)
  (:documentation "backward-popagation, return gradients to pass to preceding layer"))

(defclass layer ()
  ((weights :initarg :weights :initform nil :accessor layer-weights) ;; weights going into the layer's units
   (biases :initarg :biases :initform nil :accessor layer-biases)
   ;; per-layer activation functions (same activation function for all units in the layer)
   (activation-function :initarg :activation-function :initform #'relu :accessor layer-activation-function)
   (activation-function-derivative :initarg :activation-function-derivative :initform #'relu-derivative :accessor layer-activation-function-derivative)
   ;; use tensor-activation-function and tensor-activation-function-derivative only if you want per-unit special activation functions, they take the indicies list (multidimensional index) of a unit and return a function to use for activation
   (tensor-activation-function :initarg :tensor-activation-function :accessor layer-tensor-activation-function)
   (tensor-activation-function-derivative :initarg :tensor-activation-function-derivative :accessor layer-tensor-activation-function-derivative)))

(defclass network ()
  ((layers :initarg :layers :initform nil :accessor network-layers)
   (learning-rate :initform 0.0005 :initarg :learning-rate :accessor network-learning-rate)))

(defun make-network (&key learning-rate layers)
  (make-instance
   'network
   :layers layers
   :learning-rate (or learning-rate 0.0005)))

(defmethod network-feedforward ((n network) x)
  "x is a tensor, feedforward to the last layer and return its output"
  (with-slots (layers) n
    ;; last-out is needed to keep the output of the lastly feedforwarded layer, the output of the layers is pushed onto the list which means they are stored in reverse order, to be later popped also in "reverse" order for backprop, this means that the first entry in out-list corresponds to the output layer, this should be better than storing them in normal order because we use a list and later we need to iterate from the output of the last layer to the first layer, each entry in the list is a cons with the first element as the activated output and second as the unactivated output
    (let ((last-out x)
          (out-list nil))
      (loop for layer in layers
            do (multiple-value-bind (new-out new-out-unactivated)
                   (feedforward layer last-out)
                 (push (cons new-out new-out-unactivated) out-list)
                 (setf last-out new-out)))
      out-list)))
(defclass convolutional-layer (layer)
  ()
  (:documentation "a layer that arbitrary dimensions for its weights tensor, and convolves it with an arbitrary input tensor (convolution layer of arbitrary dimensions), the weights tensor should be 1 order higher that the input tensor, e.g. if the input tensor is a 3d tensor (image with feature maps), the shape of the weights tensor should be of 4 dimensions (4d tensor), the output tensor of the layer would be of the same order as the input tensor"))

(defun make-convolutional-layer (&key dims activation-function
                                   activation-function-derivative
                                   tensor-activation-function
                                   tensor-activation-function-derivative)
  "consutrctor for convolutional layers"
  (make-instance
   'convolutional-layer
   :weights (random-tensor dims)
   :biases (random-tensor (car dims))
   :activation-function activation-function
   :activation-function-derivative activation-function-derivative
   :tensor-activation-function tensor-activation-function
   :tensor-activation-function-derivative tensor-activation-function-derivative))

(defmethod feedforward ((l convolutional-layer) x)
  "x is an arbitrary tensor"
  (with-slots (weights biases activation-function tensor-activation-function) l
    (let* ((num-kernels (array-dimension weights 0))
           (convolution-out-size
             (mapcar
              (lambda (img-d ker-d) (- img-d (1- ker-d)))
              (array-dimensions x)
              (cdr (array-dimensions weights))))
           (out (make-array (append (list num-kernels) convolution-out-size))))
      (loop for kernel-idx from 0 below num-kernels
            do (set-array-nth
                out
                (array-map (lambda (cell) (+ cell (aref biases kernel-idx)))
                           (tensor-convolution
                            x
                            (array-nth weights kernel-idx)))
                kernel-idx))
      ;; we return 2 values, the output tensor and the unactivated output tensor
      (if tensor-activation-function
          ;; apply per-unit activation functions
          (values
           (array-map-indicies out tensor-activation-function)
           out)
          ;; apply single activation function to all units in the layer
          (values (array-map activation-function out) out)))))

(defclass 3d-convolutional-layer (convolutional-layer)
  ()
  (:documentation "a convolutional layer with 4d weights tensor and 3d input/output tensors, the depths of the input and weight tensors should be the same, this is used for convolving images with feature maps (channels)
to see the difference between this and the parent class consider the following examples:
CL-USER> (array-dimensions (let ((l (make-convolutional-layer :dims '(2 3 3 3))))
                    (feedforward l (random-tensor '(3 6 6)))))
=> (2 1 4 4)
CL-USER> (array-dimensions (let ((l (make-3d-convolutional-layer-from-dims :dims '(2 3 3 3))))
                    (feedforward l (random-tensor '(3 6 6)))))
=> (2 4 4)
"))

(defmethod feedforward :around ((l 3d-convolutional-layer) x)
  "grab the output of the parent arbitrary-convolution class, reshape it and return it, as there is always redundant dimension in the 4d tensor, this happens because the tensors (input and weights) have the same depth when doing image convolution"
  (multiple-value-bind (out unactivated-out) (call-next-method) ;; output of parent class' feedforward
    (let ((actual-convolution-out-size
            (append (list (array-dimension out 0))
                    (cdr (cdr (array-dimensions out))))))
      (values (make-array actual-convolution-out-size :displaced-to out)
              (make-array actual-convolution-out-size :displaced-to unactivated-out)))))

(defun make-3d-convolutional-layer-from-dims (&key dims activation-function activation-function-derivative tensor-activation-function tensor-activation-function-derivative)
  "consutrctor for convolutional layers"
  (make-instance
   '3d-convolutional-layer
   :activation-function activation-function
   :activation-function-derivative activation-function-derivative
   :tensor-activation-function tensor-activation-function
   :tensor-activation-function-derivative tensor-activation-function-derivative
   :weights (random-tensor dims)
   :biases (random-tensor (car dims))))

(defun make-3d-convolutional-layer (&key activation-function activation-function-derivative num-kernels kernel-depth kernel-height kernel-width tensor-activation-function tensor-activation-function-derivative)
  "consutrctor for convolutional layers"
  (make-instance
   '3d-convolutional-layer
   :activation-function activation-function
   :activation-function-derivative activation-function-derivative
   :tensor-activation-function tensor-activation-function
   :tensor-activation-function-derivative tensor-activation-function-derivative
   :weights (random-tensor (list num-kernels kernel-depth kernel-height kernel-width))
   :biases (random-tensor num-kernels)))
(defclass pooling-layer (layer)
  ((rows :initarg :rows :accessor pooling-layer-rows)
   (cols :initarg :cols :accessor pooling-layer-cols)
   (pooling-function :initarg :pooling-function :accessor pooling-layer-function)
   (unpooling-function :initarg :unpooling-function :accessor pooling-layer-unpooling-function))) ;; unpooling-function will make sense when you read later on

(defun max-pooling-function (myvec)
  (reduce #'max myvec))

(defun average-pooling-function (myvec)
  (/ (reduce #'+ myvec) (length myvec)))

(defun make-pooling-layer (&key rows cols pooling-function unpooling-function)
  (make-instance
   'pooling-layer
   :rows rows
   :cols cols
   :pooling-function (or pooling-function #'average-pooling-function)
   :unpooling-function (or unpooling-function #'average-unpooling-function)))

(defmethod feedforward ((l pooling-layer) x)
  "x is a tensor of the third order, which in case of the first layer is the actual image"
  (with-slots (rows cols pooling-function) l
    (let* ((num-channels (array-depth x))
           (img-rows (array-rows x))
           (img-cols (array-cols x))
           (out-rows (/ img-rows rows))
           (out-cols (/ img-cols cols))
           (out (make-array (list num-channels out-rows out-cols))))
      (loop for channel-idx from 0 below num-channels
            do (loop for img-row-idx from 0 below img-rows by rows
                     do (loop for img-col-idx from 0 below img-cols by cols
                              do (let ((out-row-idx (/ img-row-idx rows))
                                       (out-col-idx (/ img-col-idx cols)))
                                   (setf
                                    (aref out channel-idx out-row-idx out-col-idx)
                                    (funcall
                                     pooling-function
                                     (vectorize-array
                                      (subarray
                                       x
                                       (list channel-idx img-row-idx img-col-idx)
                                       (list rows cols)))))))))
      out)))
(defclass flatten-layer (layer) ())

(defun make-flatten-layer () (make-instance 'flatten-layer))

(defmethod feedforward ((l flatten-layer) x)
  (vectorize-array x))
(defclass dense-layer (convolutional-layer) ())

(defun make-dense-layer (&key num-units prev-layer-num-units
                           activation-function activation-function-derivative
                           tensor-activation-function tensor-activation-function-derivative)
  (make-instance
   'dense-layer
   :activation-function activation-function
   :activation-function-derivative activation-function-derivative
   :tensor-activation-function tensor-activation-function
   :tensor-activation-function-derivative tensor-activation-function-derivative
   :weights (random-tensor (list num-units prev-layer-num-units))
   :biases (random-tensor num-units)))

(defmethod feedforward :around ((l dense-layer) x)
  "return the output of the convolution, properly reshaped"
  (multiple-value-bind (out unactivated-out) (call-next-method)
    (values (make-array (list (array-dimension out 0)) :displaced-to out)
            (make-array (list (array-dimension out 0)) :displaced-to unactivated-out))))
;; (defparameter *lenet5*
;;   (make-network
;;    :layers (list
;;             (make-3d-convolutional-layer-from-dims
;;              :dims '(6 1 5 5)
;;              :activation-function #'relu
;;              :activation-function-derivative #'relu-derivative)
;;             (make-pooling-layer :rows 2 :cols 2)
;;             (make-3d-convolutional-layer-from-dims
;;              :dims '(16 6 5 5)
;;              :activation-function #'relu
;;              :activation-function-derivative #'relu-derivative)
;;             (make-pooling-layer :rows 2 :cols 2)
;;             (make-flatten-layer)
;;             (make-dense-layer :num-units 120 :prev-layer-num-units 400
;;                               :activation-function #'relu
;;                               :activation-function-derivative #'relu-derivative)
;;             (make-dense-layer :num-units 84 :prev-layer-num-units 120
;;                               :activation-function #'relu
;;                               :activation-function-derivative #'relu-derivative)
;;             (make-dense-layer :num-units 10 :prev-layer-num-units 84
;;                               :activation-function #'sigmoid
;;                               :activation-function-derivative #'sigmoid-derivative))))
(defmethod propbackward ((l convolutional-layer) layer-x layer-y layer-y-unactivated propped-deltas learning-rate)
  "compute the gradients of the layer, propped-deltas is a tensor of the errors or 'deltas' at the output nodes which is propagated back from the succeeding layer in the network, layer-x is the input image tensor that was passed to the layer during feedforwarding, layer-y is the output of the layers' feedforwarding (activation of nodes), the assumption here is that the last dimensions of weight,image tensors are equal so that the image tensor keeps its rank/order, notice that (for now) this function assumes the equality of the order of input and output tensors"
  (with-slots (weights biases
               activation-function-derivative
               tensor-activation-function-derivative)
      l
    ;; here we restore the dropped dimension, if any (a dimension is dropped if the tensors convolved have the same depth, which happens in your standard 3d convolutions with images, im not even sure why im writing code for a more general case..), we do this because its then easier to apply the math, note that all three arrays layer-y,layer-y-unactivated,propped-deltas here have the same dimensions, also note that here reshaping the arrays by adding a dimension with size 1 doesnt affect the arrays actual sizes, only their dimensionality and order/rank
    (when (not (eq (length (array-dimensions weights))
                   (length (array-dimensions layer-y))))
      (setf layer-y
            (make-array (append (list (array-dimension layer-y 0) 1)
                                (cdr (array-dimensions layer-y)))
                        :displaced-to layer-y))
      (setf layer-y-unactivated
            (make-array (append (list (array-dimension layer-y-unactivated 0) 1)
                                (cdr (array-dimensions layer-y-unactivated)))
                        :displaced-to layer-y-unactivated))
      (setf propped-deltas 
            (make-array (append (list (array-dimension propped-deltas 0) 1)
                                (cdr (array-dimensions propped-deltas)))
                        :displaced-to propped-deltas)))

    (let ((x-deltas (make-array (array-dimensions layer-x))) ;; \Delta I in the math section, the deltas at the inputs, these are to be backpropped
          (s-deltas (make-array (array-dimensions layer-y)))) ;; \Delta S in the math, the deltas "after" the activation function
      ;; compute s-deltas
      (loop for layer-y-idx from 0 below (array-size layer-y) do
        (let* ((layer-y-indicies (array-index-row-major layer-y layer-y-idx))
               (layer-y-unactivated-entry (apply #'aref (append (list layer-y-unactivated) layer-y-indicies))) ;; S^\ell[k,i,j]
               (propped-delta (apply #'aref (append (list propped-deltas) layer-y-indicies)))) ;; \Delta\hat Y[k,i,j]
          (if tensor-activation-function-derivative
              (setf (apply #'aref (append (list s-deltas) layer-y-indicies)) (* propped-delta (funcall tensor-activation-function-derivative layer-y-unactivated layer-y-indicies)))
              (setf (apply #'aref (append (list s-deltas) layer-y-indicies)) (* propped-delta (funcall activation-function-derivative layer-y-unactivated-entry))))))

      ;; compute x-deltas, this was replaced with the next sexp, im not sure if it even works but i feel like keeping it here, it is the code for the math in [[eq:old_x_delta]]
      ;; (loop for layer-x-idx from 0 below (array-size layer-x) do
      ;;   (let ((layer-x-indicies (array-index-row-major layer-x layer-x-idx)))
      ;;     (loop for layer-y-idx from 0 below (array-size layer-y) do
      ;;       (let* ((layer-y-indicies (array-index-row-major layer-y layer-y-idx))
      ;;              (s-delta (apply #'aref (append (list s-deltas) layer-y-indicies)))
      ;;              ;; see [[note:note2]], this is the pattern for the indicies of the weight to be multiplied by the entry in the input
      ;;              (weight-indicies (append (list (1- (array-dimension weights 0))) (mapcar #'+ (mapcar #'- (cdr (array-dimensions weights)) layer-x-indicies) (cdr layer-y-indicies) (make-list (length layer-x-indicies) :initial-element -1)))) ;; we add (..,-1,-1,-1) because in the math the indexing starts at 1 not 0
      ;;              (in-range t))
      ;;         ;; use in-range to check whether weight indicies are within the range of the weights tensor, again refer to [[note:note2]]
      ;;         (loop for i from 0 below (length weight-indicies) do
      ;;           (if (or (not (< (elt weight-indicies i) (array-dimension weights i)))
      ;;                   (< (elt weight-indicies i) 0))
      ;;               (setf in-range nil)))
      ;;         (when in-range
      ;;           (let* ((weight (apply #'aref (append (list weights) weight-indicies)))
      ;;                  (x-delta-to-add (* s-delta weight)))
      ;;             ;; update an x-delta
      ;;             (incf (apply #'aref (append (list x-deltas) layer-x-indicies)) x-delta-to-add)))))))

      ;; an updated solution to compute x-deltas discussed in [[any:optimization1]], first attempt
      ;; (loop for layer-x-idx from 0 below (array-size layer-x) do
      ;;   (let ((layer-x-indicies (array-index-row-major layer-x layer-x-idx)))
      ;;     (loop for weight-idx from 0 below (array-size weights) do
      ;;       (let* ((weight-indicies (array-index-row-major weights weight-idx))
      ;;              (s-delta-indicies (append
      ;;                                 (list (car weight-indicies))
      ;;                                 (mapcar #'-
      ;;                                         layer-x-indicies
      ;;                                         (cdr weight-indicies))))
      ;;              (desired-weight-indicies
      ;;                (append
      ;;                 (list (car weight-indicies))
      ;;                 (mapcar #'-
      ;;                         (cdr (array-dimensions weights))
      ;;                         (cdr weight-indicies)
      ;;                         (make-list (length (cdr weight-indicies))
      ;;                                    :initial-element 1))))
      ;;              (in-range t))
      ;;         (loop for i from 0 below (length desired-weight-indicies) do
      ;;           (if (or (not (< (elt desired-weight-indicies i)
      ;;                           (array-dimension weights i)))
      ;;                   (< (elt desired-weight-indicies i) 0))
      ;;               (setf in-range nil)))
      ;;         (loop for i from 0 below (length s-delta-indicies) do
      ;;           (if (or (not (< (elt s-delta-indicies i)
      ;;                           (array-dimension s-deltas i)))
      ;;                   (< (elt s-delta-indicies i) 0))
      ;;               (setf in-range nil)))
      ;;         (when in-range
      ;;           (let* ((weight (apply #'aref (append (list weights)
      ;;                                                desired-weight-indicies)))
      ;;                  (s-delta (apply #'aref (append (list s-deltas)
      ;;                                                 s-delta-indicies)))
      ;;                  (x-delta-to-add (* s-delta weight)))
      ;;             ;; update an x-delta
      ;;             (incf (apply #'aref (append (list x-deltas) layer-x-indicies)) x-delta-to-add)))))))

      ;; third attempt for [[any:optimization1]], here we're dropping the second-to-highest dimension because we dont need to iterate over it for every entry in layer-x, this saves us alot of iterations as it actually reduces the exponent of the time complexity (each dimension is basically another nested for loop), currently this doesnt support cases where D_K != D_I
      (let ((needed-weight-dimensions (cons (car (array-dimensions weights))
                                            (cdr (cdr (array-dimensions weights))))))
        (loop for layer-x-idx from 0 below (array-size layer-x) do
          (let ((layer-x-indicies (array-index-row-major layer-x layer-x-idx)))
            (loop for weight-idx from 0 below (reduce #'* needed-weight-dimensions) do
              (let* ((weight-indicies (from-row-major needed-weight-dimensions weight-idx))
                     (s-delta-indicies (append
                                        (cons (car weight-indicies) (cons 0 nil))
                                        (mapcar #'-
                                                (cdr layer-x-indicies)
                                                (cdr weight-indicies))))
                     (desired-weight-indicies
                       (cons
                        (car weight-indicies)
                        (mapcar #'-
                                (cdr (array-dimensions weights))
                                (cons (car layer-x-indicies) (cdr weight-indicies))
                                (make-list (length weight-indicies)
                                           :initial-element 1))))
                     (in-range t))
                (loop for i from 0 below (length desired-weight-indicies) do
                  (if (or (not (< (elt desired-weight-indicies i)
                                  (array-dimension weights i)))
                          (< (elt desired-weight-indicies i) 0))
                      (setf in-range nil)))
                (loop for i from 0 below (length s-delta-indicies) do
                  (if (or (not (< (elt s-delta-indicies i)
                                  (array-dimension s-deltas i)))
                          (< (elt s-delta-indicies i) 0))
                      (setf in-range nil)))
                (when in-range
                  (let* ((weight (apply #'aref (append (list weights)
                                                       desired-weight-indicies)))
                         (s-delta (apply #'aref (append (list s-deltas)
                                                        s-delta-indicies)))
                         (x-delta-to-add (* s-delta weight)))
                    ;; update an x-delta
                    (incf (apply #'aref (append (list x-deltas) layer-x-indicies)) x-delta-to-add))))))))

      ;; update the biases
      ;; why are we iterating through biases as if its a multidimesional array/tensor? its just a vector, this is misleading, but im leaving it this way for now
      (loop for bias-idx from 0 below (array-size biases) do
        (let ((bias-indicies (array-index-row-major biases bias-idx))
              (gradient 0)
              (needed-y-dimensions (cdr (array-dimensions layer-y))))
          (loop for layer-y-idx from 0 below (reduce #'* needed-y-dimensions) do
            (let* ((layer-y-indicies (cons (car bias-indicies)
                                           (from-row-major needed-y-dimensions layer-y-idx)))
                   (s-delta (apply #'aref (append (list s-deltas) layer-y-indicies))))
              (incf gradient s-delta)))
          ;; update bias
          (decf (apply #'aref (cons biases bias-indicies)) (* learning-rate gradient))))

      ;; update the weights
      (loop for weight-idx from 0 below (array-size weights) do
        (let ((weight-indicies (array-index-row-major weights weight-idx))
              (gradient 0)
              (needed-y-dimensions (cdr (array-dimensions layer-y))))
          ;; needed-y-dimensions are the dimensions we need to iterate through in the layer y, we dont need to iterate through the entire output as a weight is only connected to the output units that are connected to its kernel
          (loop for layer-y-idx from 0 below (reduce #'* needed-y-dimensions) do
            ;; add the kernel index to the layer-y-indicies
            (let* ((layer-y-indicies (cons (car weight-indicies) (from-row-major needed-y-dimensions layer-y-idx)))
                   (s-delta (apply #'aref (append (list s-deltas) layer-y-indicies)))
                   (i-indicies (mapcar #'+ (mapcar #'- (cdr (array-dimensions weights)) (cdr weight-indicies) (make-list (length (cdr weight-indicies)) :initial-element 1)) (cons 0 layer-y-indicies)))
                   (in-range t))
              ;; check if i-indicies are in the correct range
              (loop for i from 0 below (length i-indicies) do
                (if (or (not (< (elt i-indicies i) (array-dimension layer-x i)))
                        (< (elt i-indicies i) 0))
                    (setf in-range nil)))
              (when in-range
                ;; (print (not (member (cons i-indicies layer-y-indicies) added :test #'equal)))
                (let ((i (apply #'aref (append (list layer-x) i-indicies))))
                  (incf gradient (* s-delta i))))))
          ;; update weight
          (decf (apply #'aref (append (list weights) weight-indicies)) (* learning-rate gradient))))
      x-deltas)))
(defmethod network-propbackward ((n network) network-x network-y feedforward-out)
  "feedforward-out is the result of the network-feedforward function, its a list of cons' of out and unactivated-out, network-x and network-y should be the input and the desired output to the network, respectively"
  (with-slots (layers learning-rate) n
    (let* ((output-layer (car (car feedforward-out)))
           ;; initialize the propped deltas to (hat y - y), because we use squared error loss function
           (propped-deltas (array-map #'- output-layer network-y)))
      ;; iterate through each layer
      (loop for layer-index from (1- (length layers)) above -1 do
        ;; from the feedforward-out list, get the output of the current layer's feedforward (activated and non-activated), they are stored in reverse order so we use pop
        (let* ((mycons (pop feedforward-out))
               (layer-out (car mycons))
               (layer-unactivated-out (cdr mycons))
               (layer (elt layers layer-index))
               ;; the input to this layer is the output of the next (or previous in feedforward terms) layer, except for the first layer which receives input from the input layer which isnt in the list because its not actually a layer
               (layer-in (if (car feedforward-out) (car (car feedforward-out)) network-x)))
          ;; propbackward to the next layer, storing the deltas returned into propped-deltas to be passed onto the next layer
          (setf propped-deltas (propbackward layer layer-in layer-out layer-unactivated-out propped-deltas learning-rate)))))))
(defmethod propbackward ((l pooling-layer) layer-x layer-y layer-y-unactivated propped-deltas learning-rate)
  "a pooling layer doesnt care about layer-y,layer-y-unactivated or learning-rate, it just needs to upscale the deltas and pass them on"
  (with-slots (unpooling-function rows cols) l
    (let ((deltas (make-array (array-dimensions layer-x))))
      (loop for channel-idx from 0 below (array-depth deltas) do
        (loop for img-row-idx from 0 below (array-rows deltas) by rows do
          (loop for img-col-idx from 0 below (array-cols deltas) by cols do
            (let* ((delta-row-idx (/ img-row-idx rows))
                   (delta-col-idx (/ img-col-idx cols))
                   (img-subgrid (subarray layer-x
                                          (list channel-idx img-row-idx img-col-idx)
                                          (list rows cols)))
                   (gradient (aref propped-deltas channel-idx delta-row-idx delta-col-idx))
                   (delta-grid (funcall unpooling-function img-subgrid gradient)))
              (copy-into-array deltas delta-grid (list
                                                  channel-idx
                                                  img-row-idx
                                                  img-col-idx))))))
      deltas)))

(defun average-unpooling-function (img-subgrid gradient)
  "example usage: (progn (setf a (random-tensor '(4 4))) (average-unpooling-function a 0.7))"
  (make-array
   (array-dimensions img-subgrid)
   :initial-element (/ gradient (array-size img-subgrid))))

(defun max-unpooling-function (img-subgrid gradient)
  "example usage: (progn (setf a (random-tensor '(4 4))) (max-unpooling-function a 0.7))"
  (let* ((gradient-grid (make-array (array-dimensions img-subgrid)))
         (max-value (aref img-subgrid 0 0))
         (max-cell-indicies '(0 0)))
    (loop for row from 0 below (array-rows img-subgrid) do
      (loop for col from 0 below (array-cols img-subgrid) do
        (let ((val (aref img-subgrid row col)))
          (if (> val max-value)
              (progn
                (setf max-value val)
                (setf max-cell-indicies (list row col)))))))
    (setf (apply #'aref (append (list gradient-grid) max-cell-indicies)) gradient)
    gradient-grid))
(defmethod propbackward ((l flatten-layer) layer-x layer-y layer-y-unactivated propped-deltas learning-rate)
  "a pooling layer doesnt care about layer-y-unactivated,propped-deltas or learning-rate"
  (make-array (array-dimensions layer-x) :displaced-to propped-deltas))
(defmethod print-object ((n network) stream)
  (print-unreadable-object (n stream :type t :identity t)
    (let ((total-weights 0))
      (loop for layer in (network-layers n) do
        (when (layer-weights layer)
          (incf total-weights (array-size (layer-weights layer))))
        (format stream "~%  ~a" layer))
      (format stream
              "~&total network weights: ~a, learning rate: ~a"
              total-weights
              (network-learning-rate n)))))

(defmethod print-object ((l pooling-layer) stream)
  (print-unreadable-object (l stream :type t)
    (format stream
            "rows: ~a, columns: ~a"
            (pooling-layer-rows l)
            (pooling-layer-cols l))))

(defmethod print-object ((l convolutional-layer) stream)
  (print-unreadable-object (l stream :type t)
    (format stream
            "weights: ~a, dimensions: ~a"
            (array-size (layer-weights l))
            (array-dimensions (layer-weights l)))))
(defmethod network-full-pass ((nw network) x y)
  "do a full pass in the network, feedforward and propbackward (backpropagation)"
  (network-propbackward nw x y (network-feedforward nw x)))

(defmethod network-train ((nw network) samples &optional (epochs 1))
  "train on the given data, xs[i],ys[i] represent the input,output of the ith example, xs,ys are lists, preferrably of the simple-vector type"
  (loop for epoch from 0 below epochs do
    (loop for sample in samples
          do (let* ((x (car sample))
                    (y (cdr sample)))
               (network-full-pass nw x y)))))
(defgeneric copy (obj)
  (:documentation "make a copy of an object"))

(defmethod copy ((nw network))
  "copy a neural network"
  (make-network :layers (mapcar #'copy (network-layers nw))
                :learning-rate (network-learning-rate nw)))

(defmethod copy ((l layer))
  "copy a layer, a layer that inherits from this might have to add its own code to copy objects that arent copied by the base copy (this copy)"
  (let ((new-weights)
        (new-biases))
    (when (layer-weights l)
      (setf new-weights (make-array (array-dimensions (layer-weights l))))
      (copy-into-array new-weights (layer-weights l)))
    (when (layer-biases l)
      (setf new-biases (make-array (array-dimensions (layer-biases l))))
      (copy-into-array new-biases (layer-biases l)))
    (make-instance (type-of l)
                   :weights new-weights
                   :biases new-biases
                   :activation-function (layer-activation-function l)
                   :activation-function-derivative (layer-activation-function-derivative l))))

(defmethod copy :around ((l pooling-layer))
  "a pooling-layer has to copy more objects than the usual layer in which case the base copy function is not sufficient, this fixes that"
  (let ((new-layer (call-next-method)))
    (setf (pooling-layer-rows new-layer) (pooling-layer-rows l))
    (setf (pooling-layer-cols new-layer) (pooling-layer-cols l))
    (setf (pooling-layer-function new-layer) (pooling-layer-function l))
    (setf (pooling-layer-unpooling-function new-layer) (pooling-layer-unpooling-function l))
    new-layer))

(defun zeroize-network-weights (nw)
  "turn all the parameters of the network 0"
  (loop for layer in (network-layers nw) do
    (when (layer-weights layer)
      (setf (layer-weights layer)
            (make-array (array-dimensions (layer-weights layer)))))))

(defun add-network-weights (dest-nw src-nw)
  "add the weights of src-nw to the weights of dest-nw, src-nw has to be a copy of dest-nw"
  (loop for dest-layer in (network-layers dest-nw)
        for src-layer in (network-layers src-nw)
        do (when (layer-weights dest-layer)
             (array-map-into
              #'+
              (layer-weights dest-layer)
              (layer-weights dest-layer)
              (layer-weights src-layer)))))

(defun divide-network-weights (nw num)
  "divide all the weights of a network by num"
  (loop for layer in (network-layers nw) do
    (when (layer-weights layer)
      (array-map-into
       (lambda (weight) (/ weight num))
       (layer-weights layer)
       (layer-weights layer)))))

;; whether to terminate training or not
(defparameter *lparallel-training* nil)

(defun network-train-distributed-cpu (nw samples &optional (epochs 2))
  "samples should be conses of type simple-vector, train nw with lparallel with the number of cores your cpu has"
  (setf *lparallel-training* t)
  (let* ((nw-alias (copy nw)) ;; updates are done to nw-alias, at the end of training copied to nw
         (batch-size 10) ;; each core is gonna get that many x,y samples
         (workers (serapeum:count-cpus)) ;; set workers to number of available cpu cores
         (total-samples (length samples))
         (total-batches (floor total-samples batch-size)) ;; floor is just integer division here
         (lparallel:*kernel* (lparallel:make-kernel workers))) ;; lparallel's kernel takes care of parallelism
    (when (> (mod total-samples batch-size) 0) (incf total-batches 1))

    ;; total-rounds is the number of times we construct workers and give them each a network to train
    (loop for epoch from 0 below epochs do
      (let* ((total-rounds (floor total-batches workers))
             (channel (lparallel:make-channel))
             (batch-idx 0))
        (when (> (mod total-batches workers) 0) (incf total-rounds))
        ;; on each round we push batches to workers
        (loop for round from 0 below total-rounds while *lparallel-training* do
          (let ((active-workers 0) ;; on a round we might not need all the workers so we gotta keep track of how many workers are actually active to know how many results to ask the lparallel kernel for
                (lparallel:*task-category* 'nn)) ;; this allows for (lparallel:kill-tasks 'nn)
            (loop for worker-idx from 0 below workers do
              (when (< batch-idx total-batches)
                (let ((batch-samples (subseq samples
                                             (* batch-idx batch-size)
                                             (+ (* batch-idx batch-size) batch-size)))
                      (nw-copy (copy nw-alias)))
                  (format t "pushing batch ~a/~a~%" (1+ batch-idx) total-batches)
                  (lparallel:submit-task
                   channel
                   (lambda ()
                     (loop for sample in batch-samples while *lparallel-training* do
                       (let* ((x (car sample))
                              (y (cdr sample))
                              (out (network-feedforward nw-copy x)))
                         (network-propbackward nw-copy x y out)))
                     nw-copy)))
                (incf batch-idx 1)
                (incf active-workers 1)))
            ;; reset the nw weights to 0, as its new weights will be the averages of the copies
            (zeroize-network-weights nw-alias)
            (loop for worker-idx from 0 below active-workers do
              (let ((trained-nw-copy (lparallel:receive-result channel)))
                ;; (format t "received from worker ~a~%" worker-idx)
                (add-network-weights nw-alias trained-nw-copy)))
            ;; (format t "~a workers done~%" active-workers)
            (divide-network-weights nw-alias active-workers)
            (zeroize-network-weights nw)
            (add-network-weights nw nw-alias)))
        (format t "~%epoch done~A~%" epoch))
      (lparallel:end-kernel))))
(defmethod network-test ((nw network) samples)
  (let ((loss 0))
    (loop for sample in samples do
      (let ((x (car sample))
            (y (cdr sample)))
        (multiple-value-bind (activations unsquashed-activations)
            (network-feedforward nw x)
          (let* ((output-layer (car (car activations)))
                 (loss-to-add (reduce-array (array-map #'abs (array-map #'- output-layer y)) #'+)))
            (incf loss loss-to-add)))))
    loss))
(defun network-save-weights (nw filepath)
  "save the weight tensors of the layer of the network nw to file specified by filepath"
  (let ((weight-tensors-list))
    (with-open-file (stream filepath
                            :direction :output
                            :if-exists :supersede
                            :if-does-not-exist :create)
      (loop for layer in (network-layers nw) do
        (setf weight-tensors-list
              (append weight-tensors-list (list (layer-weights layer)))))
      (format stream "~A~%" weight-tensors-list))))

(defun network-load-weights (nw filepath)
  "load the weight tensors of a network nw from file specified by filepath"
  (let ((weight-tensors-list (with-open-file (in filepath)
                               (read in))))
    (loop for layer in (network-layers nw) do
      (let ((weight-tensor (pop weight-tensors-list)))
        (setf (layer-weights layer) weight-tensor)))))
