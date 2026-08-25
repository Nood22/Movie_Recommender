import { BrowserRouter as Router, Routes, Route, useNavigate } from "react-router-dom";
import LandingPage from "./components/LandingPage";
import TearsApp from "./components/TearsApp";
import GersApp from "./components/GersApp";

function WrapperTears() {
  const navigate = useNavigate();
  return <TearsApp goBack={() => navigate("/")} />;
}

function WrapperGers() {
  const navigate = useNavigate();
  return <GersApp goBack={() => navigate("/")} />;
}

export default function App() {
  return (
    <Router basename={process.env.PUBLIC_URL || "/"}>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/tears" element={<WrapperTears />} />
        <Route path="/gers" element={<WrapperGers />} />
      </Routes>
    </Router>
  );
}
