(defparameter *acceptor* (make-instance 'hunchentoot:easy-acceptor :port 3000))
(defun start-server (&optional (static-dir-path #p"/home/mahmooz/brain/notes/data/96/e5decf-212b-4010-85d2-475e14e276fb/"))
  "static-dir-path default to my org-attach dir"
  (setf lparallel:*kernel* (lparallel:make-kernel 1)) ;; for now we just want one core
  (setf (hunchentoot:acceptor-document-root *acceptor*) static-dir-path) ;; static dir
  (open-browser "http://127.0.0.1:3000/client.html")
  (hunchentoot:start *acceptor*)
  (sleep 10000000)) ;; cheap way of keeping program alive

(start-server ".")

(hunchentoot:define-easy-handler
    (start-training-handler :uri "/start-training"
                    :default-request-type :post)
    ()
  (setf hunchentoot:content-type* "application/json")
  (let* ((request-json (jsown:parse (hunchentoot:raw-post-data :force-text t)))
         (points (jsown:val request-json "points"))
         (epochs (jsown:val request-json "epochs"))
         (polynomial-degree (jsown:val request-json "polynomial_degree"))
         (learning-rate (jsown:val request-json "learning_rate")))
    (start-training points epochs polynomial-degree learning-rate)
    (jsown:to-json (jsown:new-js ("response" "ok")))))

(defparameter *training-objects-plist*
  '(:network nil :loss nil :samples nil :promise nil))

(ql:quickload :lparallel)

(defun start-training (points epochs polynomial-degree learning-rate)
  "points are the set of points (data) provided by the client, this starts the regression task to do a polynomial fit"
  (let ((nw (make-polynomial-regression-network polynomial-degree learning-rate))
        (samples nil))
    ;; the inputs/outputs are supposed to be vectors, just turn them into a single element vector, also points are a "3d" nested list (thats how the client stores them, its for a reason), so gotta treat them as such
    (loop for point-group in points do
      (loop for point in point-group do
        (push (cons (vector (elt point 0))
                    (vector (elt point 1)))
              samples)))
    (setf (getf *training-objects-plist* :network) nw)
    (setf (getf *training-objects-plist* :samples) samples)
    (setf (getf *training-objects-plist* :promise)
          (lparallel:future
            (loop repeat epochs do
              (network-train nw samples)
              (setf (getf *training-objects-plist* :loss) (network-test nw samples)))))))

(setf *read-default-float-format* 'double-float)

(hunchentoot:define-easy-handler
    (training-status-handler :uri "/training-status"
                            :default-request-type :get)
    ()
  (hunchentoot:no-cache)
  (setf hunchentoot:content-type* "application/json")
  (format nil (jsown:to-json (jsown:new-js
                               ;; the following line turns nil into :false, so it would work with json properly (its a trick)
                               ("done" (or (lparallel:fulfilledp (getf *training-objects-plist* :promise)) :false))
                               ("loss" (getf *training-objects-plist* :loss))))))

(hunchentoot:define-easy-handler
    (predictions :uri "/predictions"
                 :default-request-type :get)
    ()
  (hunchentoot:no-cache)
  (setf hunchentoot:content-type* "application/json")
  (let* ((nw (getf *training-objects-plist* :network))
         (samples (getf *training-objects-plist* :samples))
         (inputs (mapcar #'car samples)) ;; inputs of the samples
         (predicted-outputs (mapcar (lambda (in) (car (car (network-feedforward nw in)))) inputs)) ;; get outputs
         ;; predicted-outputs and inputs contain vectors of dimension 1, so reduce those vectors to just their first elements
         (predicted-nums (mapcar (lambda (vec) (elt vec 0)) predicted-outputs))
         (input-nums (mapcar (lambda (vec) (elt vec 0)) inputs))
         (output-points (mapcar #'list input-nums predicted-nums))) ;; join inputs and predicted outputs
    (jsown:to-json output-points)))

(hunchentoot:define-easy-handler
    (cancel-training :uri "/cancel-training"
                     :default-request-type :post)
    ()
  (hunchentoot:no-cache)
  (setf hunchentoot:content-type* "application/json")
  (lparallel:kill-tasks :default) ;; kill all lparallel tasks
  (lparallel:fulfill (getf *training-objects-plist* :promise))
  ;; (setf *training-objects-plist* '(:network nil :loss nil :samples nil :promise nil))
  (jsown:to-json (jsown:new-js ("response" "ok"))))
