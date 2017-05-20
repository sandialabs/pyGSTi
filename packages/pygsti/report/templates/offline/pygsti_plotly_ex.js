function max_width(els) {
    var ws = els.map( function(){
	return $(this).width(); }).get();
    return Math.max.apply(null, ws);
}

function max_height(els) {
    var hs = els.map( function(){
	return $(this).height(); }).get();
    return Math.max.apply(null, hs);
}

function make_wstable_resizable(id) {
    $("#" + id).resizable({
        autoHide: true,
        resize: function( event, ui ) {
            ui.element.css("padding-bottom", "7px"); //weird jqueryUI hack: to compensate for handle(?)
            ui.element.css("max-width","none"); //remove max-width
            ui.element.css("max-height","none"); //remove max-height
            var w = ui.size.width;
            var h = ui.size.height;
            ui.element.find(".dataTable").css("width",w);
            ui.element.find(".dataTable").css("height",h);
            ui.element.find(".plotContainingTD").addClass("containerAmidstResize");
            ui.element.find(".plotContainingTD").css("height",""); // so can resize freely
            console.log("Resizable table update on " + id);
        },
        stop: function( event, ui ) {
            var els = ui.element.find(".plotly-graph-div"); //want *all* plots, not just masters
            els.css("max-width","none"); //remove max-width
            els.css("max-height","none"); //remove max-height                               
            ui.element.find(".pygsti-plotgroup-master").trigger("resize");
	    var tables = ui.element.find(".dataTable");
            ui.element.css("width", max_width(tables));
            ui.element.css("height", max_height(tables));
            ui.element.find(".plotContainingTD").removeClass("containerAmidstResize");
            console.log("Resizable STOP table update on " + id);
        }
    });
}

function make_wsplot_resizable(id) {
    $("#" + id).resizable({
        autoHide: true,
        resize: function( event, ui ) {
            ui.element.css("max-width","none"); //remove max-width restriction
            ui.element.css("max-height","none"); //remove max-height restriction
            ui.element.addClass("containerAmidstResize");
        },
        stop: function( event, ui ) {
            var els = ui.element.find(".plotly-graph-div"); //want *all* plots, not just masters
            els.css("max-width","none"); //remove max-width
            els.css("max-height","none"); //remove max-height                               
            ui.element.find(".pygsti-plotgroup-master").trigger("resize");	    
            var plots = ui.element.find(".plotly-graph-div");
            ui.element.css("width", max_width(plots));
            ui.element.css("height", max_height(plots));
	    ui.element.removeClass("containerAmidstResize");
            console.log("Resizable plot update on " + id + ": " +
                        + ui.size.width + "," + ui.size.height);
        }
    });
}


function trigger_wstable_plot_creation(id) {
    $("#"+id).on("createplots", function(event) {
        var wstable = $("#"+id); //actually a div, a "workspace table"
        console.log("Initial sizing of plot table " + id + " cells");

	//0) init all plotdiv sizes (this also sets plotContainingTD class)
        wstable.find(".pygsti-plotgroup-master").trigger("init");

	//1) set widths & heights of plot-containing TDs to the
	//   "desired" or "native" values
	wstable.find("td.plotContainingTD").each( function(k,td) {
	    var plots = $(td).find(".plotly-graph-div");
	    var padding = $(td).css("padding-left");
	    if(padding == "") { padding = 0; }
	    else { padding = parseFloat(padding); }
            $(td).css("width", max_width(plots)+2*padding);
            $(td).css("height", max_height(plots)+2*padding);
	});
	
        //2) lock down initial widths of non-plot cells to they don't change later
	// (since we assume non-plot cells don't need to expand/contract).
	// We go through individual tables one-by-one, making each visible if
	// needed so that .width() and .height() will work correctly.  Also lock
	// down the heights of header (th) cells.
        wstable.children("div").each( function(k,div) {
            var was_visible = $(div).is(":visible");
            $(div).show();
            $(div).find("td").not(".plotContainingTD").each(
                function(i,el){ $(el).css("width", $(el).width()); });
	    $(div).find("th").each(
		function(i,el){ $(el).css("height", $(el).height()); });
            if(!was_visible) { $(div).hide(); }
        });

	//3) create the plots.  Each plot will look at its container's (TD's)
	//   size and size itself based on this.  If there's not enough width
	//   or height for the table, the actual width of the TDs may not be
	//   the "native" width set in 1) above [Side note: tables will *always*
	//   expand their height to their contents, ignoring any max-height or
	//   height attributes when they are too small unless you use
	//   'display: block', which isn't what we want.]
        console.log("Creating table " + id + " plots");
        wstable.find(".pygsti-plotgroup-master").trigger("create");

	//4) Thus, since the created plots may be smaller than their
	//  native sizes, and we may need to decrease table cell heights.
	wstable.find("td.plotContainingTD").each( function(k,td) {
            var plots = $(td).find(".plotly-graph-div");
	    var padding = $(td).css("padding-left");
	    if(padding == "") { padding = 0; }
	    else { padding = parseFloat(padding); }
            $(td).css("height", max_height(plots)+2*padding);
	});

	//5) It's possible that the plot div (wstable) has
        //   a max-height set.  This won't confine the table
        //   at all (since it's not a block element), so we
        //   should remove it (it would useful for plots,
        //   but not tables).
	wstable.css("max-height","none");
    });
    //Create table plots after typesetting is done and non-plot TDs are fixed
    // (note: table must be visible to compute widths correctly)
    if(typeof MathJax !== "undefined") {
	console.log("MATHJAX found - will typeset " + id + " before creating its plots.");
        MathJax.Hub.Queue(["Typeset",MathJax.Hub,id]);
        MathJax.Hub.Queue(function () {
            $("#"+id).trigger("createplots"); });
    } else {
	console.log("MATHJAX not found - creating plots for " + id + " now.");
        $("#"+id).trigger("createplots"); //trigger immediately
    }
}

function trigger_wsplot_plot_creation(id) {
    console.log("Triggering init and creation of workspace-plot" + id + " plots");
    var wsplot = $("#" + id);
    
    //1) set plot divs to their "natural" sizes
    wsplot.find(".pygsti-plotgroup-master").trigger("init");

    //2) set container (wsplot) size based on largest desired size.
    // Note: For divs, setting a max-width of 100% and a pixel value for width
    //  will cause the container to expand *up to* the pixel value (set to
    //  the "natural" width).  This seems to give good behavior, and so we
    //  set the width below.
    var plots = wsplot.find(".plotly-graph-div");
    var maxDesiredWidth = max_width(plots);
    var maxDesiredHeight = max_height(plots);
    wsplot.css("width", maxDesiredWidth);
    wsplot.css("height", maxDesiredHeight);
    console.log("Max desired size = " + maxDesiredWidth + ", " + maxDesiredHeight);

    //3) update the max-height of this container based on the maximum desired height
    existing_max_height = wsplot.css("max-height");
    if(existing_max_height != "none") {
	wsplot.css("max-height",Math.min(maxDesiredHeight,parseFloat(existing_max_height)));
    } else { wsplot.css("max-height", maxDesiredHeight); }

    //4) create the plots, based on the container size.  Note that the
    //   actual width and/or height of the container could be smaller
    //   than the values set above since both max-width and max-height
    //   work for divs (they're block elements).
    wsplot.find(".pygsti-plotgroup-master").trigger("create");

    //5) update width and/or height of container based on content, as
    // aspect ratio restrictions may have caused plots to be smaller
    // than desired, leaving free space.
    wsplot.css("width", max_width(plots));
    wsplot.css("height", max_height(plots));
    console.log("Handshake: resizing container to = " + max_width(plots) + ", " + max_height(plots));
}



function make_wsobj_autosize(boxid) {
    var timeout;
    window.addEventListener("resize",function() {
        $("#"+boxid).addClass("containerAmidstResize");
        clearTimeout(timeout);
        timeout = setTimeout(
	    function() {
		var box = $("#"+boxid);
		box.removeClass("containerAmidstResize");
		box.find(".pygsti-plotgroup-master").trigger("resize"); //always trigger masters only
		console.log("Window resize on " + boxid);
	    }, 300);
    });
}




function pex_get_container(el) {
    // first see if el lies within a TD element. If so, take this at it's container.
    var td = el.closest("td"); 
    if(td.length > 0) { return td; }

    // next see if el lies within a wsoutput group.
    var group = el.closest(".pygsti-wsoutput-group");
    if(group.length > 0) { return group; }

    return null;
}

function pex_init_plotdiv(el, natural_width, natural_height) {
    //Set initial size to the natural size
    el.css("width", natural_width);
    el.css("height",natural_height);
    
    //Set up initial maximums so plots don't get too
    // large unless we really want them to (in response
    // to a user-resizing)
    el.css("max-width",natural_width);
    el.css("max-height",natural_height);

    //Flag TDs
    var box = pex_get_container(el);
    if(box !== null && box.prop("tagName") == "TD") {
	box.addClass("plotContainingTD");
    }
}

function pex_update_plotdiv_size(el, aspect_ratio, frac_width, frac_height, orig_width, orig_height) {
    // Updates the size (width & height css) of el based on its container
    // (if a master) or directly from frac_width & frac_height (if a slave).
    // This function does *not* change the container, as el is typically a
    //  single plot and there can be multiple within a container.

    var	minfrac	= 0.0; //maybe adjust this later as a possible option?
    if(el.hasClass("pygsti-group-slave")) {
	//just set requested fractional widths and height
        fw = Math.max(minfrac, frac_width);
        fh = Math.max(minfrac, frac_height);
        ow = parseFloat(orig_width);
        oh = parseFloat(orig_height);
        el.css("width", fw*ow);
        el.css("height",fh*oh);
        //console.log("SLAVE Updating orig (" + ow + "," + oh + ") => ("
	//    + (fw*ow) + "," + (fh*oh) + ") fracs = " + fw + "," + fh);
        return;
    }
    
    //Overall strategy:
    // 1) container is expected to be setup such that it has a nonzero width()
    //    and height(), even with no content in it.  This usually means setting
    //    a pixel value for CSS 'height', and either a pixel or percentage for
    //    CSS 'width'.  See note in trigger_wsplot_plot_creation(...).
    // 2) plotly content *floats* so that its dimensions do not affect
    //    dimensions of the container
    // 3) we compute the size of the plotdiv that fits within the container
    //    and maintains the aspect ratio, if given.
    
    var w, h;
    var box = pex_get_container(el);
    var padding = 0;

    if(box.css("padding-left") != "") {
	padding = parseFloat(box.css("padding-left"));
    }

    // if no container: don't do any resizing
    if(box == null) { return;  }

    var w = box.width(); var h = box.height();    
    if(aspect_ratio == null) {
	// then just match container dimensions exactly
        el.css("width", w-2*padding);
        el.css("height",h-2*padding);
    }
    else {
        h = w / aspect_ratio; // get height corresponding to width (h may be > max-height)

	// Check if container's height (or max-height) is the
	// limiting factor given the aspect ratio (sometimes height()
	// doesn't work when not displayed)
        if(box.height() < h) { 
            h = box.height();
            w = h * aspect_ratio; //set width based on height of container
        }

	//set width based on max-height of container is unnecessary since height()
	//  should always be < max-height.
	//else if(box.css("max-height") != "none" && box.css("max-height") < h) {
	//    h = box.css("max-height");
        //    w = h * aspect_ratio; 
	//}

	//Set content dimensions
	el.css("width", w-padding);
        el.css("height",h-padding);
    }
    //console.log("pex_update_size of " + el.prop('id') + " to " + w + ", " + h + " (ratio " + aspect_ratio + ")" + " boxh = " + box.height() + " max-height = " + box.css("max-height"));
}


function pex_init_slaves(el) {
    if(el.hasClass("pygsti-group-master")) {
	var grp = el.closest(".pygsti-wsoutput-group");
	grp.find(".pygsti-group-slave").trigger("init");
	//console.log("Initializing slaves with frac = " + wfrac + "," + hfrac + " (orig = " + orig_width + "," + orig_height + ")  (cur = " + parseFloat(el.css('width')) + "," + parseFloat(el.css('height')) + ")");
    }
}

function pex_create_slaves(el, orig_width, orig_height) {
    if(el.hasClass("pygsti-group-master")) {
	var grp = el.closest(".pygsti-wsoutput-group");
	var wfrac = parseFloat(el.css('width')) / orig_width;
	var hfrac = parseFloat(el.css('height')) / orig_height;
	grp.find(".pygsti-group-slave").trigger("create", [wfrac,hfrac]);
	//console.log("Creating slaves with frac = " + wfrac + "," + hfrac + " (orig = " + orig_width + "," + orig_height + ")  (cur = " + parseFloat(el.css('width')) + "," + parseFloat(el.css('height')) + ")");
    }
}

function pex_resize_slaves(el, orig_width, orig_height) {
    if(el.hasClass("pygsti-group-master")) {
	var grp = el.closest(".pygsti-wsoutput-group");
	var wfrac = parseFloat(el.css('width')) / orig_width;
	var hfrac = parseFloat(el.css('height')) / orig_height;
	grp.find(".pygsti-group-slave").trigger("resize", [wfrac,hfrac]);
	//console.log("Resizing slaves with frac = " + wfrac + "," + hfrac + " (orig = " + orig_width + "," + orig_height + ")  (cur = " + parseFloat(el.css('width')) + "," + parseFloat(el.css('height')) + ")");
    }
}

