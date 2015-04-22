package main.java;

import java.io.Serializable;

/**
 * Metadata class stores the metadata information about the image
 * It stores x_coord, y_coord of left corner of the image in page,
 * page_number where image is taken out, page_name and 
 * line_number of the image patch in the page.
 * Metadata is made cloneable to make clone for Metadata object
 */
public class Metadata implements Serializable, Cloneable {
    
    private static final long serialVersionUID = -7311690903847009641L;	
    //X-coordinate and Y-coordinate of the left corner point of the image
    public int x_coord,y_coord;
    
    //page_name, page_number, line_number of the page from where image patch was extracted.
    public int page_number, line_number;
    public String page_name;
    
    //More parameters can be added such as pageNo, lineNo, etc...
    
    /**
     * Simple Constructor
     */
    public Metadata() {
    	x_coord = 0;
    	y_coord = 0;
		page_number = 0;
		line_number = 0;
		page_name = "";
    }
    /**
     * Perform cloning of a metadata object (Metadata class is clonable)
     */
    @Override
	public Object clone() {
		try {
			return super.clone();
		} catch (CloneNotSupportedException e) {
			throw new Error("Something impossible just happened");
		}
	}

}
