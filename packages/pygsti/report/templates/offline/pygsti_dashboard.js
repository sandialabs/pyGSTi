
$(document).ready(function() {

    window.plotman = new PlotManager();
    
    document.getElementById("defaultOpen").click();
    
    $("body").on("mousemove",function(event) {
	if (event.pageX < 50) {
            openNav();
	}
	else if (event.pageX > 250) {
            closeNav();
	}
    });
    
    
    $("figcaption").each( function() { //old: .pygsti-wsoutput-group
	$(this).on('click', function() {
	    $(this).children("span.captiondetail").toggle();}); //siblings("figcaption")
    });
    
    var elSelector		= '.header',
	elClassHidden	= 'header--hidden',
	throttleTimeout	= 500,
	$element		= $( elSelector );
    
    if( !$element.length ) return true;
    
    var $window			= $( window ),
	wHeight			= 0,
	wScrollCurrent	= 0,
	wScrollBefore	= 0,
	wScrollDiff		= 0,
	$document		= $( document ),
	dHeight			= 0,
	
	throttle = function( delay, fn ){
	    var last, deferTimer;
	    return function() {
		var context = this, args = arguments, now = +new Date;
		if( last && now < last + delay )
		{
		    clearTimeout( deferTimer );
		    deferTimer = setTimeout( function(){ last = now; fn.apply( context, args ); }, delay );
		}
		else
		{
		    last = now;
		    fn.apply( context, args );
		}
	    };
	};
    
    $window.on( 'scroll',
		throttle( throttleTimeout, function() {
		    dHeight			= $document.height();
		    wHeight			= $window.height();
		    wScrollCurrent	= $window.scrollTop();
		    wScrollDiff		= wScrollBefore - wScrollCurrent;
		    
		    if( wScrollCurrent <= 0 ) // scrolled to the very top; element sticks to the top
			$element.removeClass( elClassHidden );
		    
		    else if( wScrollDiff > 0 && $element.hasClass( elClassHidden ) ) // scrolled up; element slides in
			$element.removeClass( elClassHidden );
		    
		    else if( wScrollDiff < 0 ) // scrolled down
		    {
			if( wScrollCurrent + wHeight >= dHeight && $element.hasClass( elClassHidden ) ) // scrolled to the very bottom; element slides in
			    $element.removeClass( elClassHidden );
			
			else // scrolled down; element slides out
			    $element.addClass( elClassHidden );
		    }
		    
		    wScrollBefore = wScrollCurrent;
		}));
});


function openTab(evt, tabID) {
    var i, tabcontent, tablinks;

    // Get all elements with class="tabcontent" and hide them
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }

    // Get all elements with class="tablink" and remove the class "active"
    tablinks = document.getElementsByClassName("tablink");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }

    // Show the current tab, and add an "active" class to the button that opened the tab
    document.getElementById(tabID).style.display = "block";
    evt.currentTarget.className += " active";
    }

/* Set the width of the side navigation to 250px and the left margin of the page content to 250px */
function openNav() {
    document.getElementById("theSidenav").style.width = "250px";
    // for push: document.getElementById("main").style.marginLeft = "250px";
}

/* Set the width of the side navigation to 0 and the left margin of the page content to 0 */
function closeNav() {
    document.getElementById("theSidenav").style.width = "10px";
    // for push: document.getElementById("main").style.marginLeft = "10px";
}

