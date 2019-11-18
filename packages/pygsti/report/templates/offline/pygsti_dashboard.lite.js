function render_katex(parentEl) {
    $(parentEl).find(".math").each(function() {
        console.log("Rendering KateX");
        var texTxt = $(this).text();
        el = $(this).get(0);
        if(el.tagName == "DIV"){
            addDisp = "\\displaystyle";
        } else {
            addDisp = "";
        }
        try {
            katex.render(addDisp+texTxt, el);
        }
        catch(err) {
            $(this).html("<span class=\'err\'>"+err);
        }
    });
}


function openTab(tabID) {
    var targetTab = $(`#${tabID}`);  // hacky
    if(!targetTab.is(":visible")) {
        // Fade out open tab
        $('.tabcontent').hide().removeClass('active');
        // Remove active style on any tab links
        $('.tablink').removeClass('active');

        // Fade in target tab
        targetTab.show().addClass('active').trigger('tabchange');
        // Add active style to event target link, presumably clicked to open this tab
        $(evt.target).addClass('active');
    }
}


$(document).ready(function() {
    // Create window plot manager
    window.plotman = new PlotManager();

    // Render KaTeX
    render_katex('body');

    // Enable figure caption toggling
    $('figcaption').on('click', function() {
        // captiondetails should be divs, not spans
        $(this).children('.captiondetail').toggleClass('showcaption')
    });
});
