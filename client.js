// define canvas and its required context
var canvas = document.getElementById('sketch_canvas');
var canvasContext = canvas.getContext('2d');
var isTraining = false;

var canvasPoints = [],    /// holds strokes and points
  isCursorDown = false, /// for draw mode
  prevCursorX, prevCursorY;   /// for draw mode

// on the first click, record the first point
canvas.onmousedown = function (e) {
  /// adjust mouse position (see below)
  let pos = getEventXY(e);

  /// this is used to draw a line
  prevCursorX = pos.x;
  prevCursorY = pos.y;

  /// add new stroke
  canvasPoints.push([]);

  /// record point in this stroke
  canvasPoints[canvasPoints.length - 1].push([pos.x, pos.y]);

  /// we are in draw mode   
  isCursorDown = true;
}

canvas.onmousemove = function (e) {
  if (!isCursorDown) return;

  let pos = getEventXY(e);

  /// draw a line from previous point to this        
  canvasContext.strokeStyle = "black";
  canvasContext.beginPath();
  canvasContext.moveTo(prevCursorX, prevCursorY);
  canvasContext.lineTo(pos.x, pos.y);
  canvasContext.stroke();

  /// set previous to this point
  prevCursorX = pos.x;
  prevCursorY = pos.y;

  /// record to current stroke
  canvasPoints[canvasPoints.length - 1].push([pos.x, pos.y]);
}

canvas.onmouseup = function () {
  isCursorDown = false;
}

canvas.onmouseout = function () {
  isCusorDown = false;
}

function canvasWidth() {
  return canvas.scrollWidth;
}

function canvasHeight() {
  return canvas.scrollHeight;
}

function getEventXY(evt) {
  let rect = canvas.getBoundingClientRect();
  let x = (evt.clientX - rect.left) / (rect.right - rect.left) * canvasWidth();
  let y = (evt.clientY - rect.top) / (rect.bottom - rect.top) * canvasHeight();
  coords.innerHTML = "pos: " + x + "," + y;
  return {
    x: x,
    y: y
  };
}

function clearCanvas() {
  canvasPoints = [];
  redrawCanvas();
}

// draw X and Y axes
function drawAxes() {
  let width = canvasWidth();
  let height = canvasHeight();
  canvasContext.strokeStyle = "black";
  canvasContext.beginPath();
  // y axis
  canvasContext.moveTo(width/2, 0);
  canvasContext.lineTo(width/2, height);
  // x axis
  canvasContext.moveTo(0, height/2);
  canvasContext.lineTo(width, height/2);
  canvasContext.stroke();
}

function normalizePoints(ps) {
  let newPoints = [];
  for (let i = 0; i < ps.length; ++i) {
    let newGroupOfPoints = [];
    newPoints.push(newGroupOfPoints);
    for (let j = 0; j < ps[i].length; ++j) {
      let x = ps[i][j][0];
      let y = ps[i][j][1];
      // top-left corner is 0,0
      let normalizedX = mapNum(x, 0, canvasWidth(), -1, 1);
      let normalizedY = mapNum(y, canvasHeight(), 0, 1, -1);
      newGroupOfPoints.push([normalizedX, normalizedY])
    }
  }
  return newPoints;
}

// unlike normalizePoints, this takes a one-dimensional array, not a nested one, this is because the server returns the predicted samples as a 1d array
function denormalizePoints(ps) {
  let newPoints = [];
  for (let i = 0; i < ps.length; ++i) {
    let p = ps[i];
    let x = p[0];
    let y = p[1];
    let denormalizedX = mapNum(x, -1, 1, 0, canvasWidth());
    let denormalizedY = mapNum(y, 1, -1, canvasHeight(), 0);
    let newPoint = [denormalizedX, denormalizedY];
    newPoints.push(newPoint);
  }
  return newPoints;
}

// see my note [[id:4adab93c-126d-4ecf-8a31-16b32fdc06f1][linearly mapping an interval onto another]]
function mapNum(x, srcMin, srcMax, destMin, destMax) {
  return (x * destMax - x * destMin - srcMin * destMax + destMin * srcMax) /
    (srcMax - srcMin);
}

function startTraining() {
  if (isTraining) {
    alert("already started training!");
    return;
  }
  clearLog();
  isTraining = true;
  let epochs = parseInt(document.getElementById("epochs_input").value);
  let degree = parseInt(document.getElementById("degree_input").value);
  let learningRate = parseFloat(document.getElementById("learning_rate_input").value);
  if (isNaN(epochs)) {
    epochs = 100000;
  }
  if (isNaN(degree)) {
    degree = 1;
  }
  if (isNaN(learningRate)) {
    learningRate = 0.001;
  }
  post("/start-training", {
    "points": normalizePoints(canvasPoints),
    "epochs": epochs,
    "polynomial_degree": degree,
    "learning_rate": learningRate,
  });
  let statusInterval = setInterval(function() {
    get("/training-status", {}, function(responseText) {
      let responseJson = JSON.parse(responseText);
      if (responseJson.done) {
        logMessage("done training");
        clearInterval(statusInterval);
        isTraining = false;
      } else {
        logMessage(`loss: ${responseJson.loss}\n`);
      }
    })
    post
  }, 40);
  let predictionsInterval = setInterval(function() {
    get("/predictions", {}, function(responseText) {
      if (isTraining) {
        let responseJson = JSON.parse(responseText);
        let pts = denormalizePoints(responseJson); // response json is basically a list of points
        redrawCanvas();
        drawPoints(pts, "blue");
      } else {
        clearInterval(predictionsInterval);
      }
    })
    post
  }, 200);
}

function addEpochs() {
  if (isTraining) {
    alert("you cant add epochs while training");
    return;
  }
}

function logMessage(msg) {
  console.log(msg);
  let textareaElement = document.getElementById('log_textarea');
  textareaElement.innerHTML += msg;
  textareaElement.scrollTop = textareaElement.scrollHeight;
}

function clearLog() {
  document.getElementById('log_textarea').innerHTML = "";
}

// send a post request, formats data to json, return the json respones
function post(url, data) {
  var xhr = new XMLHttpRequest();
  xhr.open("POST", url, true);
  xhr.setRequestHeader('Content-Type', 'application/json');
  xhr.send(JSON.stringify(data));
}

function get(url, params, callback) {
  var xhr = new XMLHttpRequest();
  xhr.open("GET", url + formatParams(params), true);
  xhr.onreadystatechange = function() {
    if (xhr.readyState == 4 && xhr.status) {
      callback(xhr.responseText);
    }
  }
  xhr.send(null);
}
// from https://stackoverflow.com/questions/8064691/how-do-i-pass-along-variables-with-xmlhttprequest
function formatParams(params) {
  if (Object.keys(params).length === 0)
    return "";
  return "?" + Object
    .keys(params)
    .map(function (key) {
      return key + "=" + encodeURIComponent(params[key])
    })
    .join("&")
}

function drawPoints(pts, strokeStyle="black") {
  /// draw a line from previous point to this        
  let tmpPrevX = pts[0][0];
  let tmpPrevY = pts[0][1];
  canvasContext.strokeStyle = strokeStyle;
  canvasContext.beginPath();
  canvasContext.moveTo(tmpPrevX, tmpPrevY);
  pts.forEach(function(point) {
    let currentX = point[0];
    let currentY = point[1];
    canvasContext.lineTo(currentX, currentY);
    canvasContext.stroke();
    tmpPrevX = currentX;
    tmpPrevY = currentY;
  });
}

function redrawCanvas() {
  let rect = canvas.getBoundingClientRect();
  canvasContext.clearRect(0, 0, rect.width, rect.height);
  drawAxes();
  canvasPoints.forEach(function (groupOfPoints) {
    drawPoints(groupOfPoints);
  });
}

function cancelTraining() {
  if (!isTraining) {
    alert("no training in progress!");
    return;
  }
  isTraining = false;
  post("/cancel-training", {});
}

window.onload = function() {
  redrawCanvas();
}