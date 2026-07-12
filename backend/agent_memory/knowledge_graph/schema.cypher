// Agent memory Neo4j schema — apply via neo4j-migrate, NOT app code.
// Safe to re-run (IF NOT EXISTS).

CREATE CONSTRAINT user_id IF NOT EXISTS
FOR (u:User) REQUIRE u.id IS UNIQUE;

CREATE CONSTRAINT entity_key IF NOT EXISTS
FOR (e:Entity) REQUIRE (e.user_id, e.name) IS UNIQUE;

CREATE CONSTRAINT fact_key IF NOT EXISTS
FOR (f:UserFact) REQUIRE (f.user_id, f.text) IS UNIQUE;

CREATE CONSTRAINT constraint_key IF NOT EXISTS
FOR (c:Constraint) REQUIRE (c.user_id, c.text) IS UNIQUE;
