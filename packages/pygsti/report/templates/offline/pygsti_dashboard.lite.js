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
