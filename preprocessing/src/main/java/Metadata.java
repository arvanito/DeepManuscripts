package main.java;

import java.io.Serializable;

/**
 *
 */
public class Metadata implements Serializable, Cloneable {
    
    private static final long serialVersionUID = -7311690903847009641L;	
    public int x_coord,y_coord;
	public int page_number, line_number;
	public String page_name;
    
    //More parameters can be added such as pageNo, lineNo, etc...
    
    // Null Constructor
    public Metadata() {
    	x_coord = 0;
    	y_coord = 0;
		page_number = 0;
		line_number = 0;
		page_name = "";
    }

	@Override
	public Object clone() {
		try {
			return super.clone();
		} catch (CloneNotSupportedException e) {
			throw new Error("Something impossible just happened");
		}
	}

}
