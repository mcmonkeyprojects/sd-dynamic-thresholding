let dynthres_update_enabled = function() {
    return Array.from(arguments);
};

(function(){
    let accordions = {};
    let enabled = {};
    onUiUpdate(() => {
        let accordion_id_prefix = "#dynthres_";
        let extension_checkbox_class = ".dynthres-enabled";

        dynthres_update_enabled = function() {
            let res = Array.from(arguments);
            let tabname = res[1] ? "img2img" : "txt2img";

            let checkbox = accordions[tabname]?.querySelector(extension_checkbox_class + ' input');
            checkbox?.dispatchEvent(new Event('change'));

            return res;
        };

        function attachEnabledButtonListener(checkbox, accordion) {
            let span = accordion.querySelector('.label-wrap span');
            let badge = document.createElement('input');
            badge.type = "checkbox";
            badge.checked = checkbox.checked;
            badge.addEventListener('click', (e) => {
                checkbox.checked = !checkbox.checked;
                badge.checked = checkbox.checked;
                checkbox.dispatchEvent(new Event('change'));
                e.stopPropagation();
            });

            badge.className = checkbox.className;
            badge.classList.add('primary');
            span.insertBefore(badge, span.firstChild);
            let space = document.createElement('span');
            space.innerHTML = "&nbsp;";
            span.insertBefore(space, badge.nextSibling);

            checkbox.addEventListener('change', () => {
                let badge = accordion.querySelector('.label-wrap span input');
                badge.checked = checkbox.checked;
            });
            checkbox.parentNode.style.display = "none";
        }

        if (Object.keys(accordions).length < 2) {
            let accordion = gradioApp().querySelector(accordion_id_prefix + 'txt2img');
            if (accordion) {
                accordions.txt2img = accordion;
            }
            accordion = gradioApp().querySelector(accordion_id_prefix + 'img2img');
            if (accordion) {
                accordions.img2img = accordion;
            }
        }

        if (Object.keys(accordions).length > 0 && accordions.txt2img && !enabled.txt2img) {
            enabled.txt2img = accordions.txt2img.querySelector(extension_checkbox_class + ' input');
            attachEnabledButtonListener(enabled.txt2img, accordions.txt2img);
        }
        if (Object.keys(accordions).length > 0 && accordions.img2img && !enabled.img2img) {
            enabled.img2img = accordions.img2img.querySelector(extension_checkbox_class + ' input');
            attachEnabledButtonListener(enabled.img2img, accordions.img2img);
        }
    });
})();
