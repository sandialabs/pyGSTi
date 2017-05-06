
function pex_get_container(el) {
    // first see if el lies within a TD element. If so, take this at it's container.
    var td = el.closest("td"); 
    if(td.length > 0) { return td; }

    // next see if el lies within a wsoutput group.
    var group = el.closest(".pygsti-wsoutput-group");
    if(group.length > 0) { return group; }

    return null;
}

function pex_init_container(el, natural_width, natural_height) {
    //initialize (always) native-sized plotholder div
    var box = pex_get_container(el);

    //Set up initial maximums so plots don't get too
    // large unless we really want them to (in response
    // to a user-resizing)
    el.css("max-width",natural_width);
    el.css("max-height",natural_height);
  
    if(box !== null) {
	if(box.prop("tagName") == "TD") {
	    box.css("width",natural_width);
	    box.css("height",natural_height);
	    box.addClass("plotContainingTD");
	}
	else {
	    //box.css("display","table-cell");
	    box.css("width",natural_width);
	    box.css("height",natural_height);
	    existing_max_height = box.css("max-height");
	    if(existing_max_height != "none") {
		box.css("max-height",Math.min(natural_height,parseFloat(existing_max_height)));
	    }
	    else { box.css("max-height", natural_height); }
	    //box.css("max-width",natural_width); // often set at 100% to make
	    //  width function as a "desired" width -- height doesn't do this,
	    //  and so we can set it.
	}
    }
}


function pex_update_size(el, aspect_ratio) {
    //Overall strategy:
    // 1) container is expected to be setup such that its width will 
    //    automatically expand as desired (usually up to some "natural width").
    //    (td: set width of td directly, div: set max-width only, as block divs
    //     naturally expand in width)
    // 2) plotly content *floats* so that its dimensions do not affect
    //    dimensions of the container
    // 3) we must then set the height of the container to match either the
    //    natural height (if no aspect ratio) or the height which matches the
    //    present width of the container (if locking aspect ratio).
    // 4) The container may have a max-height set, so we check to see if our
    //    setting of the container height actually stuck -- if not, use the
    //    *actual* height  (and correspondign width if locking aspect ratio)
    //    to set the content dimensions.
    var w, h;
    var box = pex_get_container(el);

    // if no container: don't do any resizing
    if(box == null) { return;  }

    var w = box.width(); var h = box.height();    
    if(aspect_ratio == null) {
	// then just match smaller of container and natural dimensions
        el.css("width", w);
        el.css("height",h);
    }
    else {
        h = w / aspect_ratio; // get height corresponding to width (h may be > max-height)
	box.css("height",h); //set container height based on its width

	// Check if container's actual height can be h
	// (or if a user-supplied max-height restriction limits)
        if(box.height() < h) { 
            h = box.height();
            w = h * aspect_ratio; //set width based on max-height of container
	    box.css("width",w);   //reset container width based on its max-height
	    // (note: containter's width indicates the biggest we ever want the
	    //  figure to get automatically - and here we're altering that based on
	    //  an imposed max height)
        }

	//Finally, set content dimensions
	el.css("width", w);
        el.css("height",h);
    }
    //console.log("pex_update_size of " + el.prop('id') + " to " + w + ", " + h + " (ratio " + aspect_ratio + ")");
}
