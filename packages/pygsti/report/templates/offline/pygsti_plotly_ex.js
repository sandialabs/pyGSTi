
function pex_get_container(el) {
    // first see if el lies within a TD element. If so, take this at it's container.
    var td = el.closest("td"); 
    if(td.length > 0) { return td; }

    // next see if el lies within a wsoutput group.
    var group = el.closest(".pygsti-wsoutput-group");
    if(group.length > 0) { return group; }

    var box = null;

    // UNUSED: TODO REMOVE
    // finally, take first parent with max-height or max-width set
    //el.parents().each(function(i,e) {
    //	if(box == null && ($(e).css("max-height") != "none"
    //			   || $(e).css("max-width") != "none")) {
    //	    box=$(e)
    //	}
    //});
    return box;
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


function pex_update_size(el, container, natural_width, natural_height, aspect_ratio) {
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
    
    // if natural height is given, then place into container and see
    //   whether container will expand to allow this.
    //if(natural_width != null) { w = natural_width; } //Math.min(w,natural_width); }
    //if(natural_height != null) { h = natural_height; } //Math.min(h,natural_height); }
    //el.css("width", w); el.css("height",h);

    //first check width
    //if(box.width() < w) { // Check if container can be width w (max-width restriction limits?)
    //	w = box.width();
    //}
    //if(box.height() < h) {
    //	h = box.height();
    //}
    
    // set w,h as smaller of container and natural dimensions
    //var w = box.width(); var h = box.height();
    //if(natural_width != null) { w = Math.min(w,natural_width); }
    //if(natural_height != null) { h = Math.min(h,natural_height); }
    //var w1 = w, h1 = h, h2 = null, w2=null;
    
    if(aspect_ratio == null) {
	// then just match smaller of container and natural dimensions
        el.css("width", w);
        el.css("height",h);
    }
    else {
        h = w / aspect_ratio; // get height corresponding to width (h may be > max-height)
	//h2 = h;
        //el.css("width", w); // adjust el based on box width
        //el.css("height",h); // (this will "inflate" container's height, unless box is a TD)

	box.css("height",h); //set container height based on its width

	//if(box.prop("tagName") == "TD") {
	//    //In case where box is a TD the plot is decoupled (floats) from the container, so
	//    // setting el's dims will *not* inflate box.  Behavior of TD's width (set initially
	//    // to the "natural" width) will automatically inflate width of TD as desired, but
	//    // height of TDs will override table's height - so need to set TD's height to just
	//    // what we need.
	//    box.css("height",h);
	//}

	// Check if container's actual height can be h
	// (or if a user-supplied max-height restriction limits)
        if(box.height() < h) { 
            h = box.height();
            w = h * aspect_ratio; //set width based on max-height of container
	    box.css("width",w);   //reset container width based on its max-height
	    // (note: containter's width indicates the biggest we ever want the
	    //  figure to get automatically - and here we're altering that based on
	    //  an imposed max height)

	    //update TD height again
	    //if(box.prop("tagName") == "TD") { box.css("height",h); }
	    // (commented out b/c max-height *should* override height so unnecessary)
        }

	//Finally, set content dimensions
	el.css("width", w);
        el.css("height",h);
    }
    console.log("pex_update_size to " + w + ", " + h + " (ratio " + aspect_ratio + ")");
    //" natural " + natural_width + "," + natural_height +
    //console.log(" w1= " + w1 + ", w2=" + w2 + ", h1=" + h1 + ", h2=" + h2);
}




//$(".relwrap").each(function(i,el) {
//  $(el).css("background-color","green");
//  var contentWidth = $(el).children("div").first().width();
//  var contentHeight = $(el).children("div").first().height();
//  var p = $(el).parents("td").first();
//  p.css("width",contentWidth);
//  p.css("height",contentHeight);
//  console.log("Adding dims: " + contentWidth + "," + contentHeight
//   + " to id=" + p.attr("id"));
//});
