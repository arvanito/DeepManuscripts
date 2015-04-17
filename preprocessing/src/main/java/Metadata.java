package main.java;

import java.io.Serializable;


public class Metadata implements Serializable {
    
    private static final long serialVersionUID = -7311690903847009641L;	
    public int x_coord,y_coord;
    public int width, height;
    
    //More parameters can be added such as pageNo, lineNo, etc...
    
    // Null Constructor
    public Metadata() {
    	x_coord = (Integer) null;
    	y_coord = (Integer) null;
    	width = (Integer) null;
    	height = (Integer) null;
    }
    
    //Simple Constructor
    public Metadata(int x, int y, int w, int h) {
    	x_coord = x;
    	y_coord = y;
    	width = w;
    	height = h;
    }
    
        

}
