
function pex_get_container(el) {
    var box = null;  //take as reference first parent with max-height or max-width set
    el.parents().each(function(i,e) {
    	if(box == null && ($(e).css("max-height") != "none"
			   || $(e).css("max-width") != "none")) {
	    box=$(e)
	}
    });
    return box;
}

function pex_update_size(el, container, natural_width, natural_height, aspect_ratio) {
    var w, h;
    var box = (container == null) ? pex_get_container(el) : container;

    if(box == null) {
	// no container: just use original dimensions if provided,
	//  otherwise take the immediate parent as the container
	if(natural_width != null && natural_height != null) {
	    el.css("width", natural_width);
	    el.css("height",natural_height);
	    //console.log("Resize to natural" + natural_width + ", " + natural_height);
	    return;
	}
	else { box = el.parent(); }
    }

    // set w,h as smaller of container and natural dimensions
    var w = box.width(); var h = box.height();
    if(natural_width != null) { w = Math.min(w,natural_width); }
    if(natural_height != null) { h = Math.min(h,natural_height); }
    var w1 = w, h1 = h, h2 = null, w2=null;
    
    if(aspect_ratio == null) {
	// then just match smaller of container and natural dimensions
        el.css("width", w);
        el.css("height",h);
    }
    else {
        h = w / aspect_ratio; // get height corresponding to width (h may be > max-height)
	h2 = h;
        el.css("width", w); // adjust el based on box width
        el.css("height",h); // (this will "inflate" container's height)
        if(box.height() < h) { // Check if container can be height h (or if max-height restriction limits)
            h = box.height();
            w = h * aspect_ratio; //set width based on max-height of container
	    w2 = w;
            el.css("width", w);
            el.css("height",h); 
        }
    }
    console.log("pex_update_size to " + w + ", " + h +
		" (ratio " + aspect_ratio + " natural " + natural_width + "," + natural_height + ")");
    console.log(" w1= " + w1 + ", w2=" + w2 + ", h1=" + h1 + ", h2=" + h2);
}
