(function(){
var accordions = {};
var enabled = {};
onUiUpdate(() => {
    var accordion_id_prefix = "#dynthres_";
    var extension_checkbox_class = ".dynthres-enabled";

    function triggerEvent(element, eventName) {
        var event = document.createEvent("HTMLEvents");
        event.initEvent(eventName, false, true);
        element.dispatchEvent(event);
    }

    function updateActiveState(checkbox, accordion) {
        // change checkbox state
        const badge = accordion.querySelector('.label-wrap span input');
        badge.checked = checkbox.checked;
    }

    function attachEnabledButtonListener(checkbox, accordion) {
        // add checkbox
        const span = accordion.querySelector('.label-wrap span');
        const badge = document.createElement('input');
        badge.type = "checkbox";
        badge.checked = checkbox.checked;
        badge.addEventListener('click', (e) => {
            checkbox.checked = !checkbox.checked;
            triggerEvent(checkbox, 'change');
            e.stopPropagation();
        });

        badge.className = checkbox.className;
        badge.classList.add('primary');
        span.insertBefore(badge, span.firstChild);
        var space = document.createElement('span');
        space.innerHTML = "&nbsp;";
        span.insertBefore(space, badge.nextSibling);

        checkbox.addEventListener('click', () => {
            updateActiveState(checkbox, accordion);
        });
    }

    if (Object.keys(accordions).length < 2) {
        var accordion = gradioApp().querySelector(accordion_id_prefix + 'txt2img');
        if (accordion)
            accordions.txt2img = accordion;
        accordion = gradioApp().querySelector(accordion_id_prefix + 'img2img');
        if (accordion)
            accordions.img2img = accordion;
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
