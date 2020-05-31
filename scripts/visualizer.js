(function() {
    const default_fov = 120;
    let panoramas = [
        { "id": 'panorama',
          "num": 0,
          "panorama": document.getElementById('panorama'),
          "src_folder": 'results',
          "src_name": '_output.jpg'
        }, {
            "id": 'panorama2',
            "num": 1,
            "panorama": document.getElementById('panorama2'),
            "src_folder": 'data',
            "src_name": '_content.jpg'
        }];
    let viewers = [null, null];
    
    let selector = document.getElementById('selector');
    let cur_img = 1;
    const shift = {'prev': -1, 'next': 1};
    const max_img = 7;

    let view_original = 0;
    const view = ['hidden', 'visible'];
    let type = document.getElementById('type');
    const type_vals = ['original', 'stylized'];

    function updatePanorama() {
        panoramas.forEach(panorama => {
            panorama['panorama'].style.width = `${window.innerWidth}px`;
            panorama['panorama'].style.height = `${window.innerHeight}px`;
            viewers[panorama['num']] = pannellum.viewer(panorama['id'], {
                "type": "equirectangular",
                "panorama": `${panorama['src_folder']}/${cur_img}${panorama['src_name']}`,
                "autoLoad": true,
                "autoRotate": 2,
                "showControls": false
            });
            viewers[panorama['num']].setHfov(default_fov, false);
        });
        view_original = 0;
        let original = panoramas[1]['panorama'];
        original.style.visibility = 'hidden';
    }

    function swapPanorama() {
        let prev_viewer = viewers[view_original];
        updateView();
        let original = panoramas[1]['panorama'];
        original.style.visibility = view[view_original];
        type.textContent = type_vals[view_original];
        let top_viewer = viewers[view_original];
        let hov = prev_viewer.getHfov();
        let pitch = prev_viewer.getPitch();
        let yaw = prev_viewer.getYaw();
        let auto_rotate = prev_viewer.getConfig().autoRotate;
        if (!auto_rotate) {
            top_viewer.stopAutoRotate()
        }
        top_viewer.setHfov(hov, false);
        top_viewer.setPitch(pitch, false);
        top_viewer.setYaw(yaw, false);
    }

    function updateView() {
        view_original = (view_original + 1) % 2;
    }
    
    function updateImg(cmd) {
        cur_img += shift[cmd];
        checkImg();
        displayImg();
    }
    
    function displayImg() {
        selector.src = `results/${cur_img}_output.jpg`;
    }

    function checkImg() {
        if (cur_img < 1) {
            cur_img = max_img;
        }
        if (cur_img > max_img) {
            cur_img = 1;
        }
    }

    updatePanorama();
    window.addEventListener("resize", updatePanorama);

    window.addEventListener("keydown", e => {
        //Space
        console.log(e);
        if (e.keyCode == 32) {
            swapPanorama();
        }
    });
})();
