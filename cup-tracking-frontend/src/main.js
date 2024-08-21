const videoElement = document.createElement('video');
videoElement.autoplay = true;
document.body.appendChild(videoElement);

// access the user's webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        videoElement.srcObject = stream;
        videoElement.play();
        videoElement.addEventListener('loadeddata', sendFrameToServer);
    })
    .catch((error) => {
        console.error('Error accessing webcam: ', error);
    });

function isVideoPlaying(video) {
    return !!(video.currentTime > 0 && !video.paused && !video.ended && video.readyState > 2);
}

function sendFrameToServer() {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');

    if (canvas.width === 0 || canvas.height === 0) {
        console.error('Canvas dimensions are zero. Ensure videoElement is loaded.');
        return;
    }

    setInterval(() => {
        if (isVideoPlaying(videoElement)) {
            ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
            const dataURL = canvas.toDataURL('image/jpeg');

            if (dataURL === 'data:,') {
                console.error('Data URL is empty. Canvas content might not be properly captured.');
                return;
            }
            
            fetch('http://localhost:8000/process_frame/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ frame: dataURL }),
            })
                .then(response => response.json())
                .then(data => {
                    console.log(data.message);
                    if (data.keypoints) {
                        drawKeypoints(ctx, data.keypoints);
                    }
                })
                .catch(error => console.error('Error:', error));
        } else {
            console.error('Video is not playing. Please ensure the video is loaded and playing.');
        }
    }, 1000 / 30); // 30 FPS
}

function drawKeypoints(ctx, keypoints) {
    ctx.fillStyle = 'red';
    keypoints.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
        ctx.fill();
    });
}