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

function get_wsobj_group(id) {
    var obj = $("#" + id);
    if(obj.hasClass("pygsti-wsoutput-group")) {
        return obj; // then id was the id of the entire group (OK)
    } else { // assume id was for one of the items within a group
        return obj.closest(".pygsti-wsoutput-group");
    }
}

function make_wstable_resizable(id) {
    wsgroup = get_wsobj_group(id);
    if( wsgroup.hasClass('ui-resizable')) return; //already make resizable
    
    wsgroup.resizable({
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
    wsgroup = get_wsobj_group(id);
    if( wsgroup.hasClass('ui-resizable')) return; //already make resizable
    
    wsgroup.resizable({
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


function trigger_wstable_plot_creation(id, initial_autosize) {
    $("#"+id).on("createplots", function(event) {
        var wstable = $("#"+id); //actually a div, a "workspace table"
        console.log("Initial sizing of plot table " + id + " cells");

	//0) init all plotdiv sizes (this also sets plotContainingTD class)
        wstable.find(".pygsti-plotgroup-master").trigger("init");

	//0.5) set the widths of non-plot-containing TDs whose contents are short/simple 
	//     (no children) - this keeps "Gx" columns, e.g. from being too large since
	//     setting the other column widths can for some reason make their widths larger too(!)
	var fnForSingleTableDivs = function(k,div) {
            var was_visible = $(div).css('display') != 'none'; //is(":visible");
            $(div).show();
            $(div).find("td").not(".plotContainingTD").each(
                function(i,el){
		    if($(el).children().length == 1) {
			$(el).css("width", $(el).width()); } });
	    $(div).find("th").each(
		function(i,el){
		    if($(el).children().length == 1) {
			$(el).css("height", $(el).height()); } });
            if(!was_visible) { $(div).hide(); }
        }
	if(wstable.hasClass("single_switched_value")) {
	    fnForSingleTableDivs( 0, wstable[0]);
	} else {
            wstable.children("div").each( fnForSingleTableDivs );
	}


	//1) set widths & heights of plot-containing TDs to the
	//   "desired" or "native" values
	TDtoCheck = null; // the TD element used to check for width settling
	wstable.find("td.plotContainingTD").each( function(k,td) {
	    var plots = $(td).find(".plotly-graph-div");
	    var padding = $(td).css("padding-left");
	    if(padding == "") { padding = 0; }
	    else { padding = parseFloat(padding); }
	    desiredW =	max_width(plots)+2*padding;
            desiredH =	max_height(plots)+2*padding
            $(td).css("width", desiredW);
            $(td).css("height", desiredH);
	    if(TDtoCheck === null) TDtoCheck = $(td); //just take the first one

	    if(!initial_autosize) {
		firstChild = $(td).children("div.pygsti-wsoutput-group").first()
		firstChild.css("width", max_width(plots)+2*padding);  // these don't seem to do 
		firstChild.css("height", max_height(plots)+2*padding);// anything... (?)
		//console.log("DEBUG: SETTING width/height CSS of first child! ID=" + firstChild.attr("id"))
		//console.log("Vals = ",max_width(plots)+2*padding, max_height(plots)+2*padding)
		$(td).css("min-width", desiredW); //makes TD's width actually work
	    }
	});

	//1.5) wait for the computed widths of the table cells (just one representative
	//  TDtoCheck cell currently) to settle.  In some browser, Firefox in ptic,
	//  it takes some time for the browser to respond to the desired withs set above
	//  and update the widths of the TD elements in the table.  The code below
	//  waits for this settling to occur before proceeding to step 2.
	last_w = 0; cnt = 0; 
	nSettle = 2; //number of times we need to get the same width to call it "settled"
        var intervalFn = setInterval( function() {
	    if(TDtoCheck === null) {
		clearInterval(intervalFn);
		wstable.trigger("after_widths_settle");
	    }
	    else {
		w = TDtoCheck.width();
		if(last_w == parseFloat(w)) {
		    if(cnt < 2) cnt += 1;
		    else {
			clearInterval(intervalFn);
			wstable.trigger("after_widths_settle");
		    }
		}
		else last_w = parseFloat(w);
	    }
	}, 200);
    });
    
    $("#"+id).on("after_widths_settle", function(event) {
	var wstable = $("#"+id); //actually a div, a "workspace table"                            
	console.log("Widths settled; Creating table " + id + " plots");

        //2) lock down initial widths of non-plot cells so they don't change later
	// (since we assume non-plot cells don't need to expand/contract).
	// We go through individual tables one-by-one, making each visible if
	// needed so that .width() and .height() will work correctly.  Also lock
	// down the heights of header (th) cells.
	var tab = wstable.closest(".tabcontent")
	var parent_was_visible = true;
	if(tab.length > 0 && tab.css('display') == 'none') { //!tab.is(":visible")) {
	    parent_was_visible = false;
	    tab.show();
	}

	var fnForSingleTableDivs = function(k,div) {
            var was_visible = $(div).css('display') != 'none'; //is(":visible");
            $(div).show();
            $(div).find("td").not(".plotContainingTD").each(
                function(i,el){ $(el).css("width", $(el).width()); });
	    $(div).find("th").each(
		function(i,el){ $(el).css("height", $(el).height()); });
            if(!was_visible) { $(div).hide(); }
        }

	if(wstable.hasClass("single_switched_value")) {
	    fnForSingleTableDivs( 0, wstable[0]);
	} else {
            wstable.children("div").each( fnForSingleTableDivs );
	}

	if(!parent_was_visible) { tab.hide(); } 

	//3) create the plots.  Each plot will look at its container's (TD's)
	//   size and size itself based on this.  If there's not enough width
	//   or height for the table, the actual width of the TDs may not be
	//   the "native" width set in 1) above [Side note: tables will *always*
	//   expand their height to their contents, ignoring any max-height or
	//   height attributes when they are too small unless you use
	//   'display: block', which isn't what we want.]
        //console.log("Creating table " + id + " plots");
        wstable.find(".pygsti-plotgroup-master").trigger("create");

	//Finish the rest of plot creation *after* all the plots have been 
	// created (the trigger stmt above just queues them).
	if(plotman != null) {
	    plotman.enqueue( function() {
		$("#"+id).trigger("after_createplots");
	    }, "Finishing table " + id + " plot creation");
	} else {
	    wstable.trigger("after_createplots");
	}

    });

    $("#"+id).on("after_createplots", function(event) {
	console.log("Post plot-creation sizing of plot table " + id + " cells");
	var wstable = $("#"+id); //actually a div, a "workspace table"

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
        //   should remove it (it would be useful for plots,
        //   but not tables).
	wstable.css("max-height","none");

	// 6) If this table is within a <figure> tag try to set
	//    caption detail's width based on rendered table width
	caption = wstable.closest('figure').children('figcaption:first');
	if(caption.length > 0) {
            caption.css('width', Math.round(wstable.width()*0.9) + 'px');
	}

	// 7) Remove the hard-set width and height of the plot-containing
	//   div used to prevent initial auto-sizing.
	if(!initial_autosize) {
	    wstable.find("td.plotContainingTD").each( function(k,td) {
		firstChild = $(td).children("div.pygsti-wsoutput-group").first()
		firstChild.css("width", "");
		firstChild.css("height", "");
	    });
	}
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

function trigger_wsplot_plot_creation(id, initial_autosize) {
    console.log("Triggering init and creation of workspace-plot" + id + " plots");
    var wsplot = $("#" + id);
    
    //1) set plot divs to their "natural" sizes
    wsplot.find(".pygsti-plotgroup-master").trigger("init");

    //2) set container (wsplot) size based on largest desired size.
    // Note: For divs, setting a max-width of 100% and a pixel value for width
    //  will cause the container to expand *up to* the pixel value (set to
    //  the "natural" width).  This seems to give good behavior, and so we
    //  set the width below.
    var wsplotgroup = null;
    if(wsplot.hasClass("pygsti-wsoutput-group")) {
        wsplotgroup = wsplot; // then id was the id of the entire group (OK)
    } else { // assume id was for one of the plots within a group
             // (also OK, but we want to set the css of the *group* div below)
        wsplotgroup = wsplot.closest(".pygsti-wsoutput-group");
    }

    if(!initial_autosize) {
	//ignore max-width and max-height settings to desired height
	wsplotgroup.css("max-width","none");
	wsplotgroup.css("max-height","none");
    }

    var plots = wsplotgroup.find(".plotly-graph-div");
    var maxDesiredWidth = max_width(plots);
    var maxDesiredHeight = max_height(plots);
    wsplotgroup.css("width", maxDesiredWidth);
    wsplotgroup.css("height", maxDesiredHeight);
    console.log("Max desired size = " + maxDesiredWidth + ", " + maxDesiredHeight);

    //3) update the max-height of this container based on the maximum desired height
    existing_max_height = wsplotgroup.css("max-height");
    if(existing_max_height != "none") {
	wsplotgroup.css("max-height",Math.min(maxDesiredHeight,parseFloat(existing_max_height)));
    } else { wsplotgroup.css("max-height", maxDesiredHeight); }

    //4) create the plots, based on the container size.  Note that the
    //   actual width and/or height of the container could be smaller
    //   than the values set above since both max-width and max-height
    //   work for divs (they're block elements).
    wsplot.find(".pygsti-plotgroup-master").trigger("create");

    //5) update width and/or height of container based on content, as
    // aspect ratio restrictions may have caused plots to be smaller
    // than desired, leaving free space.
    wsplotgroup.css("width", max_width(plots));
    wsplotgroup.css("height", max_height(plots));
    console.log("Handshake: resizing container to = " + max_width(plots) + ", " + max_height(plots));

    // 6) If this table is within a <figure> tag try to set
    //    caption detail's width based on rendered table width
    caption = wsplotgroup.closest('figure').children('figcaption:first');
    if(caption.length > 0) {
        caption.css('width', Math.round(wsplotgroup.width()*0.9) + 'px');
    }

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
    
    console.log("Calling pex_init_plotdiv with width ", natural_width, "and height ", natural_height);
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

    // Note: for box.width() and .height() to work below (and therefore for
    //  all the processing to work correctly), the element at hand needs to
    //  be *visible* -- if it's not, a table cell (for instance) will have
    //  the width given by CSS and not the computed width give the table's
    //  total width.
    var tab = el.closest(".tabcontent")
    var val_container = el.closest(".single_switched_value")
    var in_invisible_tab = false;
    var in_invisible_val = false;
    if(tab.length > 0 && tab.css('display') == 'none') { //!tab.is(":visible")) {
	in_invisible_tab = true;
	tab.show();
    }
    if(val_container.length > 0 && val_container.css('display') == 'none') { // !val_container.is(":visible")) {
	in_invisible_val = true;
	val_container.show();
    }
    
    var w = box.width(); var h = box.height();
    //console.log("box width", box.width());
    //console.log("box height", box.height());
    //console.log("aspect ratio", aspect_ratio);    
    if(aspect_ratio == null) {
	// then just match container dimensions exactly
        el.css("width", w-2*padding);
        el.css("height",h-2*padding);
    }
    else {
        h = w / aspect_ratio; // get height corresponding to width (h may be > max-height)
        //console.log("h ", h);
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
    
    if(in_invisible_tab) { tab.hide(); }
    if(in_invisible_val) { val_container.hide(); } 
    //console.log("pex_update_size of " + el.prop('id') + " to " + w + ", " + h + " (ratio " + aspect_ratio + ")" + " boxh = " + box.height() + " max-height = " + box.css("max-height") + " container(" + box.prop('id') + ").width() = " + box.width());
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


function PlotManager(){
    this.queue = [];
    this.labelqueue = [];
    this.busy = false;
    this.processor = null;
}

PlotManager.prototype.run = function(){
    console.log("PLOTMANAGER: starting run queue execution");
    
    if (this.processor !== null) {
	clearInterval(processor);
    }
    $("#status").show();
    this.processor = setInterval(function(pm) {
	if (!pm.busy) {
	    pm.busy = true;
	    if (pm.queue.length){
		var label = pm.labelqueue.shift(); //pop();
		var callback = pm.queue.shift(); //pop();
		$("#status").text(label + " (" + pm.queue.length + " remaining)");
		console.log("PLOTMANAGER: " + label + " (" + pm.queue.length + " remaining)");
		try {
		    callback();
		} finally {
		    pm.busy = false; // in case an error occurs, don't block queue
		}
	    }
	    else {
		pm.busy = false;
	    }
	}
	if (pm.queue.length <= 0) {
	    console.log("PLOTMANAGER: queue empty!");
            clearInterval(pm.processor);
	    pm.processor = null;
            $("#status").hide();
	}
    }, 200, this); //pass this as "pm" argument to function
}

PlotManager.prototype.enqueue = function(callback, label, autostart=true){
    this.queue.push(callback);
    this.labelqueue.push(label);
    if(autostart && this.processor === null) { this.run(); } // in case queue hasn't started
}


function cloneAndReplace(jquery_element) {    
    var old_element = jquery_element.get(0);
    var new_element = old_element.cloneNode(true);
    old_element.parentNode.replaceChild(new_element, old_element);
    return $(new_element)
}

function cloneAndAppend(jquery_element) {    
    var old_element = jquery_element.get(0);
    var new_element = old_element.cloneNode(true);
    old_element.parentNode.appendChild(new_element, old_element);
    return $(new_element)
}



/*
CollapsibleLists.js
An object allowing lists to dynamically expand and collapse

Created by Stephen Morley - http://code.stephenmorley.org/ - and released under
the terms of the CC0 1.0 Universal legal code:

http://creativecommons.org/publicdomain/zero/1.0/legalcode
*/

const CollapsibleLists = (function(){
  // Makes all lists with the class 'collapsibleList' collapsible. The
  // parameter is:
  //
  // doNotRecurse - true if sub-lists should not be made collapsible
  function apply(doNotRecurse){
    [].forEach.call(document.getElementsByTagName('ul'), node => {
      if (node.classList.contains('collapsibleList')){
        applyTo(node, true);
        if (!doNotRecurse){
          [].forEach.call(node.getElementsByTagName('ul'), subnode => {
            subnode.classList.add('collapsibleList')
          });
        }
      }
    })
  }

  // Makes the specified list collapsible. The parameters are:
  //
  // node         - the list element
  // doNotRecurse - true if sub-lists should not be made collapsible
  function applyTo(node, doNotRecurse){
    [].forEach.call(node.getElementsByTagName('li'), li => {
      if (!doNotRecurse || node === li.parentNode){
        li.style.userSelect       = 'none';
        li.style.MozUserSelect    = 'none';
        li.style.msUserSelect     = 'none';
        li.style.WebkitUserSelect = 'none';
        li.addEventListener('click', handleClick.bind(null, li));
        toggle(li);
      }
    });
  }

  // Handles a click. The parameter is:
  // node - the node for which clicks are being handled
  function handleClick(node, e){
    let li = e.target;
    while (li.nodeName !== 'LI'){
      li = li.parentNode;
    }
    if (li === node){
      toggle(node);
    }
  }

  // Opens or closes the unordered list elements directly within the
  // specified node. The parameter is:
  //
  // node - the node containing the unordered list elements
  function toggle(node){
    const open = node.classList.contains('collapsibleListClosed');
    const uls  = node.getElementsByTagName('ul');
    [].forEach.call(uls, ul => {
      let li = ul;
      while (li.nodeName !== 'LI'){
        li = li.parentNode;
      }
      if (li === node){
        ul.style.display = (open ? 'block' : 'none');
      }
    });

    node.classList.remove('collapsibleListOpen');
    node.classList.remove('collapsibleListClosed');
    if (uls.length > 0){
      node.classList.add('collapsibleList' + (open ? 'Open' : 'Closed'));
    }
  }
  return {apply, applyTo};
})();
