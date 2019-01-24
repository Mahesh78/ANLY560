import java.sql.*;

public class sakila {

	public static void main(String[] args) throws SQLException{

		Connection conn = null;
		Statement stmt = null;
		ResultSet res = null;
		
		try {
			conn = DriverManager.getConnection("jdbc:mysql://localhost:3306/sakila", "user" , "pass");
			
			stmt = conn.createStatement();
			
			res = stmt.executeQuery("SELECT a.title, a.description, c.first_name, c.last_name FROM sakila.film a LEFT JOIN sakila.film_actor b ON a.film_id = b.film_id LEFT JOIN sakila.actor c ON c.actor_id = b.actor_id WHERE a.title LIKE 'm%'");
			
			while (res.next()) {
				System.out.println(res.getString("a.title") + "	" + res.getString("a.description"));
			}
	}
		catch (Exception ex) {
			ex.printStackTrace();
}
		finally {
			if (res != null) {
				res.close();
			}
			if (conn != null) {
				conn.close();
			}
		}
}
}