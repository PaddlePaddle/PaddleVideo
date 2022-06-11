// 获取上传按钮和进度条
var uploadButtons = document.querySelectorAll(".upload-button");
var uploadButton = uploadButtons[1];
console.log(uploadButtons)
var progressBar = document.querySelector(".upload-button:nth-child(2) .progress-bar");

// 进度条完成时的宽度
let width = uploadButton.getBoundingClientRect().width;
// 假定上传时间为5s
let uploadTime = 5000;

const dropArea = document.querySelector('body');
const dropBox = document.querySelector('.drag-area');
dragText = dropBox.querySelector('p');
button = dropBox.querySelector('.upload');
input = dropBox.querySelector('input');
let file;
// let videoTag=[];
button.onclick = () => {
    input.click();
}

function storeFile() {
    let fileType = file.type;
    let validExtensions = ['video/mp4', 'video/mov', "video/ogg"];
    if (validExtensions.includes(fileType)) {
        upload();
        dropArea.classList.remove('active');
        let fileReader = new FileReader();
        fileReader.onload = () => {
            let fileURL = fileReader.result;
            console.log(fileURL);
            videoTag = `<video width="320" height="240" controls><source src="${fileURL}" type="video/mp4">您的浏览器不支持Video标签。</video>`;
        }
        fileReader.readAsDataURL(file);
    } else {
        alert('这不是视频哦~');
        button.style.opacity = "1";
        dropArea.classList.remove('active');
    }
}

input.addEventListener('change', function () {
    file = this.files[0];
    storeFile();
})
dropArea.addEventListener('dragover', (event) => {
    event.preventDefault();
    uploadButton.style.opacity = "0";
    uploadButton.classList.remove("uploaded");
    dropArea.classList.add('active');
    dragText.textContent = '释放来新建';
    button.style.opacity = "1"
});
dropArea.addEventListener('dragleave', () => {
    dropArea.classList.remove('active');
    dragText.innerHTML = "拖放来新建<br>或者";
    button.style.opacity = "1"
});
dropArea.addEventListener('drop', (event) => {
    event.preventDefault();
    file = event.dataTransfer.files[0];
    storeFile();
})

function upload() {
    dragText.textContent = '正在处理';
    dropArea.classList.add('active');
    uploadButton.style.opacity = "1";
    console.log('uploadButton', uploadButton)

    // 先移除之前的完成样式
    uploadButton.classList.remove("uploaded");

    //设置正在上传.uploading样式
    uploadButton.classList.add("uploading");

    //假设5秒后上传完成
    setTimeout(() => {
        uploadButton.classList.replace("uploading", "uploaded");
    }, uploadTime);

    let start = null;

    function grow(timestamp) {
        // 动画开始时的时间戳
        if (!start) start = timestamp;
        // 距离开始时已经过的时间戳
        let progress = timestamp - start;
        //按比例增加进度条宽度
        progressBar.style.width = `${Math.min(
            width * (progress / uploadTime),
            width
        )}px`;

        // 如果上传未完成，继续执行此函数，递归循环
        if (progress < uploadTime) {
            window.requestAnimationFrame(grow);
        }
        if (progress >= uploadTime) {
            dragText.textContent = '处理完成';
            dropArea.classList.add('active');
        }
    }


    // 开始执行grow函数
    window.requestAnimationFrame(grow);
}